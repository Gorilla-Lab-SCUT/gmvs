import argparse
import copy
import os

import torch
import torch.nn.functional as F
import numpy as np
import open3d as o3d
from tqdm import tqdm

import prim3d
from gmvs.lib.build import planar_initer as _C


def remove_isolate_component_by_diameter(
    o3d_mesh,
    diameter_percentage: float = 0.05,
    keep_mesh: bool = False,
    remove_unreferenced_vertices: bool = True,
):
    import copy

    assert diameter_percentage >= 0.0
    assert diameter_percentage <= 1.0
    max_bb = o3d_mesh.get_max_bound()
    min_bb = o3d_mesh.get_min_bound()
    size_bb = np.abs(max_bb - min_bb)
    filter_diag = diameter_percentage * np.linalg.norm(size_bb)

    vertices = np.asarray(o3d_mesh.vertices)
    faces = np.asarray(o3d_mesh.triangles)

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        triangle_clusters, cluster_n_triangles, _ = o3d_mesh.cluster_connected_triangles()
    triangle_clusters = np.asarray(triangle_clusters) + 1
    cluster_n_triangles = np.asarray(cluster_n_triangles)

    largest_cluster_idx = cluster_n_triangles.argmax() + 1
    for idx in range(1, len(cluster_n_triangles) + 1):  # set label 0 to keep
        if idx == largest_cluster_idx:  # must keep the largest
            triangle_clusters[triangle_clusters == idx] = 0
        else:
            cluster_triangle = triangle_clusters == idx
            cluster_index = np.unique(faces[cluster_triangle])
            cluster_vertices = vertices[cluster_index]
            cluster_bbox = np.abs(
                np.amax(cluster_vertices, axis=0) - np.amin(cluster_vertices, axis=0)
            )
            cluster_bbox_diag = np.linalg.norm(cluster_bbox, ord=2)
            if cluster_bbox_diag >= filter_diag:
                triangle_clusters[triangle_clusters == idx] = 0
    mesh_temp = copy.deepcopy(o3d_mesh) if keep_mesh else o3d_mesh
    mesh_temp.remove_triangles_by_mask(triangle_clusters > 0.5)

    if remove_unreferenced_vertices:
        mesh_temp.remove_unreferenced_vertices()
    return mesh_temp


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument("--depth_normal_dir", type=str, default="results/0616_00/mvs/depth_normal/")
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/ScanNet/manhattan_data/reornder_name/",
        help="directory to store the origin data",
    )
    parser.add_argument(
        "--superpixel_dir",
        "-spd",
        type=str,
        help="directory to store the superpixel results",
    )
    parser.add_argument(
        "--save_dir",
        "-sd",
        type=str,
        default="data/ScanNet/manhattan_data/0616_00/render_mask/",
        help="directory to store the output results",
    )
    parser.add_argument(
        "--fliter_angle_thresh",
        type=float,
        default=70,
        help="save visualization",
    )
    parser.add_argument(
        "--clean_mesh",
        action="store_true",
        help="save visualization",
    )
    parser.add_argument(
        "--clean_mesh_percentage",
        type=float,
        default=0.05,
        help="save visualization",
    )
    parser.add_argument(
        "--gen_mask",
        action="store_true",
        help="save visualization",
    )
    parser.add_argument(
        "--mask_dir",
        type=str,
        default="planar_mask",
        help="save visualization",
    )
    parser.add_argument("--height", type=int, default=480, help="the height of input images")
    parser.add_argument("--width", type=int, default=640, help="the width of input images")
    parser.add_argument(
        "--vis",
        action="store_true",
        help="save visualization",
    )
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()
    depth_normal_dir = args.depth_normal_dir
    data_dir = args.data_dir
    superpixel_dir = args.superpixel_dir
    save_dir = args.save_dir
    height = args.height
    width = args.width
    device = "cuda" if torch.cuda.is_available() else "cpu"

    import time
    tic_ = time.time()
    # -------------------------------------------------------
    ''' Hyper Parameters '''
    # -------------------------------------------------------
    poisson_mesh_thresh = clean_pts_thresh = 0.03
    poisson_depth = 9
    FUSION_THRESH = int(height * width / 32)
    # -------------------------------------------------------
    NTHEADS = 64
    NONPLANAR_PERCENT = 0.75
    COSSIM_THRESH = 0.8
    MATCH_RATIO_THRESH = 0.75
    MAX_FILTER_DEPTH = 10
    # -------------------------------------------------------

    os.makedirs(save_dir, exist_ok=True)
    if args.gen_mask:
        os.makedirs(os.path.join(save_dir, f"../../{args.mask_dir}"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "planar_normal_clean"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "depth_normal_filter_clean"), exist_ok=True)
    if args.vis:
        os.makedirs(os.path.join(save_dir, "vis_clean"), exist_ok=True)

    # get pose files
    pose_files_list = os.listdir(os.path.join(data_dir, "pose"))
    pose_files_list.sort(key=lambda x: int(x.split(".")[0]))
    num_imgs = len(pose_files_list)

    # get mvs depthnormal file
    depthnorm_files_list = os.listdir(depth_normal_dir)
    depthnorm_files_list.sort(key=lambda x: int(x.split(".")[0]))
    assert num_imgs == len(depthnorm_files_list)


    # load all poses and depthnormals
    all_poses = [
        torch.from_numpy(
            np.loadtxt(
                os.path.join(data_dir, "pose", c2wf)
            )
        ).float() 
        for c2wf in tqdm(pose_files_list, desc="loading poses")
    ]
    all_depthsnormals = [
        torch.from_numpy(
            np.load(
                os.path.join(depth_normal_dir, dnf)
            )
        ).float() 
        for dnf in tqdm(depthnorm_files_list, desc="loading depth normal")
    ]
    all_poses = torch.stack(all_poses).to(device)
    all_depthsnormals = torch.stack(all_depthsnormals).to(device)

    # get direction
    intrinsic = torch.from_numpy(
        np.loadtxt(os.path.join(data_dir, "intrinsic.txt"))
    ).float()
    fx = intrinsic[0, 0]
    fy = intrinsic[1, 1]
    cx = intrinsic[0, 2]
    cy = intrinsic[1, 2]

    yy, xx = torch.meshgrid(
        torch.arange(height, dtype=torch.float32),
        torch.arange(width, dtype=torch.float32),
        indexing="ij",
    )
    xx = (xx - cx) / fx
    yy = (yy - cy) / fy
    zz = torch.ones_like(xx)

    dirs = torch.stack((xx, yy, zz), axis=-1)  # OpenCV convention
    dirs_ = dirs.reshape(-1, 3).to(device)     # [H * W, 3]
    
    ##########################################################
    # get mvs points
    ##########################################################
    points_all = []
    points_all_filter = []
    viewnorm_thresh = np.cos(float(args.fliter_angle_thresh) / 180 * np.pi)
    for idx, extrinsic in tqdm(enumerate(all_poses),
        desc="Gather mvs points"
    ):
        origins = extrinsic[:3, 3]  # [3]
        origins = torch.tile(origins, (height * width, 1))  # [H * W, 3]
        rot_mat = extrinsic[:3, :3]
        dirs = dirs_ @ (rot_mat.T)  # [H * W, 3]

        depthnorm = all_depthsnormals[idx]

        depth = depthnorm[..., :1].reshape(-1, 1)
        normal = depthnorm[..., 1:].reshape(-1, 3)
        depth_mask = (depth > 0).squeeze()
        points = origins[depth_mask] + depth[depth_mask] * dirs[depth_mask]

        viewnorm_mask = ((-dirs[depth_mask]) * normal[depth_mask]).sum(-1) > viewnorm_thresh

        # Save the filtered depth normal
        depth_normal_mask = torch.where(depth_mask == True, 1, 0)
        depth_normal_mask[torch.where(depth_normal_mask > 0.5)] = viewnorm_mask.to(depth_normal_mask.dtype)
        depth_normal_mask = (
            depth_normal_mask[..., None].expand(-1, 4).reshape(height, width, 4)
        )
        depthnorm_filter = torch.where(depth_normal_mask > 0.5, depthnorm, torch.Tensor([0.]).to(device))
        all_depthsnormals[idx] = depthnorm_filter
        
        pointsn = torch.cat((points, normal[depth_mask]), dim=1)
        points_all.append(pointsn.cpu().numpy())
        points_all_filter.append(pointsn[viewnorm_mask].cpu().numpy())

    # cat all pts
    points_all_filter = np.concatenate(points_all_filter)
    if args.vis:
        points_all = np.concatenate(points_all)
        # save mvs points
        np.savetxt(
            os.path.join(save_dir, "mvs_depthnorm.xyz"),
            points_all,
            fmt="%.4f",
        )
        np.savetxt(
            os.path.join(save_dir, "mvs_depthnorm_filter.xyz"),
            points_all_filter,
            fmt="%.4f",
        )
    len_filter_pts = len(points_all_filter)
    assert len_filter_pts == (all_depthsnormals[...,0]>0.).sum().cpu().numpy()
    del points_all
    
    ##########################################################
    # pts to mesh
    ##########################################################
    ptsn = points_all_filter.astype(np.float32)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(ptsn[:, :3])
    pcd.normals = o3d.utility.Vector3dVector(ptsn[:, 3:])

    with o3d.utility.VerbosityContextManager(o3d.utility.VerbosityLevel.Debug) as cm:
        (
            mesh_poisson,
            densities,
        ) = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=poisson_depth)
    o3d.io.write_triangle_mesh(os.path.join(save_dir, "poisson_mesh.ply"), mesh_poisson)

    # filter the poisson mesh
    mesh_poisson_points = o3d.geometry.PointCloud(mesh_poisson.vertices)
    dist = mesh_poisson_points.compute_point_cloud_distance(pcd)
    dist = np.asarray(dist)
    ind = np.where(dist > poisson_mesh_thresh)[0]
    mesh_poisson.remove_vertices_by_index(ind)
    o3d.io.write_triangle_mesh(os.path.join(save_dir, "poisson_mesh_filter.ply"), mesh_poisson)

    #########################################################
    # clean the mesh again to remove isolate faces
    # mesh to render raycasting scene
    #########################################################
    if args.clean_mesh:
        mesh_poisson = remove_isolate_component_by_diameter(
            mesh_poisson, diameter_percentage=float(args.clean_mesh_percentage)
        )
        o3d.io.write_triangle_mesh(os.path.join(save_dir, "filter_clean.ply"), mesh_poisson)

        mesh_sampled_pts = o3d.geometry.PointCloud(mesh_poisson.vertices)
        dist = pcd.compute_point_cloud_distance(mesh_sampled_pts)  
        dist = np.asarray(dist)  
        assert len_filter_pts == len(dist)
        filter_pts_clean_mask = torch.from_numpy(
            dist < clean_pts_thresh
        ).to(device)
        print(f"Cleaned depth normal : {filter_pts_clean_mask.sum()} / {len(filter_pts_clean_mask)}")

        all_depthsnormals_mask = (all_depthsnormals[..., 0] > 0).reshape(-1)
        all_depthsnormals_mask[torch.where(all_depthsnormals_mask)] = filter_pts_clean_mask
        all_depthsnormals_mask = all_depthsnormals_mask.reshape(-1, height, width)
        all_depthsnormals[torch.where(~all_depthsnormals_mask)] = 0.
        if args.vis:
            points_all_filter_clean = points_all_filter[filter_pts_clean_mask.cpu().numpy()]
            # save mvs points
            np.savetxt(
                os.path.join(save_dir, "mvs_depthnorm_filter_clean.xyz"),
                points_all_filter_clean,
                fmt="%.4f",
            )
            del points_all_filter_clean
        del all_depthsnormals_mask
    del points_all_filter

    ##########################################################
    # Render and fusion mask 
    ##########################################################
    vertices = np.asarray(mesh_poisson.vertices)
    faces = np.asarray(mesh_poisson.triangles)
    vertices_rc = torch.from_numpy(vertices).float().cuda()
    faces_rc = torch.from_numpy(faces).to(torch.int32).cuda()
    RT = prim3d.libPrim3D.create_raycaster(vertices_rc, faces_rc)
    torch.cuda.empty_cache()

    pbar = tqdm(enumerate(zip(pose_files_list, all_poses)), total=num_imgs)
    for idx, (pose_file, extrinsic) in pbar:
        pbar.set_description(f"({idx + 1}/{num_imgs}) Render planar depth and normal")
        name = pose_file[:-4]

        # save the filtered and cleaned depth normal
        depthnorm = all_depthsnormals[idx]
        np.save(
            os.path.join(save_dir, "depth_normal_filter_clean", f"{name}.npy"),
            depthnorm.float().cpu().numpy(),
        )

        # get the camera extrinsic to get the view direction for ray casting
        origins = extrinsic[:3, 3]  # [3]
        origins = torch.tile(origins, (height * width, 1))  # [H * W, 3]
        rot_mat = extrinsic[:3, :3]
        dirs = dirs_ @ (rot_mat.T)  # [H * W, 3]

        # ray casting to get the depth&normal from reconstructed mesh
        num_rays = origins.shape[0]
        filter_normal = torch.zeros_like(origins)
        filter_depth = torch.zeros([num_rays], dtype=torch.float32, device="cuda")
        primitive_ids = torch.zeros([num_rays], dtype=torch.int32, device="cuda") - 1
        RT.invoke(origins, dirs, filter_depth, filter_normal, primitive_ids)
        filter_normal = F.normalize(filter_normal, p=2, dim=-1).clamp(-1.0, 1.0)
        filter_depth = filter_depth.reshape(height, width)
        filter_normal = filter_normal.reshape(height, width, 3)
        filter_depth[filter_depth > MAX_FILTER_DEPTH] = 0
        filter_normal[filter_depth > MAX_FILTER_DEPTH] = 0
        filter_depth[torch.where(filter_depth < 0.05)] = 0
        np.save(os.path.join(save_dir, "planar_normal_clean", f"{name}.npy"), filter_normal.cpu().numpy())
        
        filter_depth = filter_depth.cpu().numpy()
        filter_normal = filter_normal.cpu().numpy()
        planar_mask = np.where(filter_depth<0.01, 1, 0)
        seg_ids = np.load(
                os.path.join(superpixel_dir, f"segid_npy/{name}.npy")
            ).astype(np.int32)
           
        if args.gen_mask:
            fusion_planar_mask = _C.fusion_planar_mask(
                seg_ids,
                planar_mask,
                filter_normal,
                height,
                width,
                FUSION_THRESH,
                NONPLANAR_PERCENT,
                COSSIM_THRESH,
                MATCH_RATIO_THRESH,
            )
            # keep the plane prior region
            seg_ids = np.where(
                (fusion_planar_mask > 300), fusion_planar_mask, seg_ids
            )  # for mvs empty area, use vps normal
            fusion_planar_mask[fusion_planar_mask == 300] = 0  # for objects segs, remove them
            np.save(
                os.path.join(save_dir, f"../../{args.mask_dir}", f"{name}.npy"), fusion_planar_mask
            )
        else:
            fusion_planar_mask = np.zeros_like(planar_mask)

        if args.vis:
            import copy
            import cv2

            # read image
            img = cv2.imread(os.path.join(data_dir, "images", f"{name}.png"))
            # vis the planar mask image
            planar_img = img.copy()
            planar_img = np.where(planar_mask[..., None].repeat(3, axis=-1) > 0.5, img, 0)
            # vis the superpixel and fusion planar mask image
            color_bar = np.random.randint(
                0, 255, [max(seg_ids.max(), fusion_planar_mask.max()) + 1, 3]
            )
            color_bar[0] = 0
            seg_ids_rgb = color_bar[seg_ids]
            fusion_planar_mask_rgb = color_bar[fusion_planar_mask]

            fusion_rgb_vsp = copy.deepcopy(fusion_planar_mask_rgb)
            fusion_rgb_vsp[fusion_planar_mask > 300] = 255
            fusion_rgb_planar = copy.deepcopy(fusion_planar_mask_rgb)
            fusion_rgb_planar[fusion_planar_mask >= 128] = 255

            fusion_rgb = np.where(fusion_planar_mask[..., None].repeat(3, axis=-1) > 0, img, 0)
            # vis depth&normal
            normal_rgb = ((filter_normal + 1) * 127.5).astype(np.uint8)
            output_img = np.concatenate(
                [
                    np.concatenate(
                        [img, planar_img, fusion_planar_mask_rgb, fusion_rgb_vsp], axis=1
                    ),
                    np.concatenate(
                        [seg_ids_rgb, fusion_rgb, normal_rgb, fusion_rgb_planar], axis=1
                    ),
                ],
                axis=0,
            )
            vis_path = os.path.join(save_dir, "vis_clean", f"{name}.png")
            cv2.imwrite(vis_path, output_img)

    toc_ = time.time() - tic_
    print("Total time : ", toc_)