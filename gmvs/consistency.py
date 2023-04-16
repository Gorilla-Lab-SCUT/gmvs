# Copyright (c) Gorilla-Lab. All rights reserved.
import os
from typing import List, Optional, Tuple

import cv2
import numpy as np
import torch
from tqdm import tqdm

from .primitives import Problem


def read_camera(cam_path: str) -> Tuple[List[float]]:
    with open(cam_path, "r") as f:
        content = f.read()
    token = content.split()
    extrins = np.array([float(_t) for _t in token[1:17]]).reshape(4, 4)
    R = extrins[:3, :3].reshape(9).tolist()
    t = extrins[:3, 3].tolist()
    K = [float(_t) for _t in token[18:27]]

    depth_min_max = [float(token[27]), float(token[30])]

    return (K, R, t, depth_min_max)


def run_patch_match(
    result_dir: str,
    output_dir: str,
    problems: List[Problem],
    verbose: bool = False,
    first_stage: bool = True,
) -> Tuple[torch.Tensor]:
    import gmvs.src as _C

    num_imgs = len(problems)
    pbar = tqdm(range(num_imgs), total=num_imgs)

    cpp_problems = []
    cameras = []
    gray_images = []
    depths = []
    normals = []
    costs = []
    for i in pbar:
        pbar.set_description(
            f"({i + 1}/{num_imgs}): " f"Process image {problems[i].ref_image_id:04} ..."
        )
        problem = problems[i]
        ref_image_id = problem.ref_image_id
        image_path = os.path.join(
            result_dir, "images", f"{ref_image_id:04}.jpg")
        camera_path = os.path.join(
            result_dir, "cams", f"{ref_image_id:04}_cam.txt")
        save_dir = os.path.join(output_dir, f"{ref_image_id:04}")
        depth_path = os.path.join(
            save_dir,
            "depth.npy" if first_stage else "depth_geom.npy",
        )
        normal_path = os.path.join(
            save_dir,
            "normal.npy" if first_stage else "normal_geom.npy",
        )
        cost_path = os.path.join(
            save_dir,
            "cost.npy" if first_stage else "cost_geom.npy",
        )
        gray_image = (
            torch.from_numpy(cv2.imread(
                image_path, cv2.IMREAD_GRAYSCALE)).float()
        )  # [H, W, 3]
        depth = torch.from_numpy(np.load(depth_path))  # [H, W]
        normal = torch.from_numpy(np.load(normal_path))  # [H, W, 3]
        cost = torch.from_numpy(np.load(cost_path))  # [H, W]
        height, width = depth.shape
        (K, R, t, (depth_min, depth_max)) = read_camera(camera_path)
        camera = _C.load_camera(K, R, t, height, width, depth_min, depth_max)
        cpp_problems.append(problem._to_cpp())
        cameras.append(camera)
        gray_images.append(gray_image)
        depths.append(depth)
        normals.append(normal)
        costs.append(cost)

    # init patch matcher
    PatchMatcher = _C.PatchMatcher()
    PatchMatcher.params.geom_consistency = True
    PatchMatcher.params.max_iterations = 2
    PatchMatcher.params.max_iterations = 2

    gray_images = torch.stack(gray_images)  # [num_images, H, W]
    depths = torch.stack(depths)  # [num_images, H, W]
    normals = torch.stack(normals)  # [num_images, H, W, 3]
    costs = torch.stack(costs)  # [num_images, H, W]

    PatchMatcher.add_samples(
        cpp_problems,
        cameras,
        gray_images,
        depths,
        normals,
        costs)

    output_depths = torch.zeros_like(depths)
    output_normals = torch.zeros_like(normals)
    output_costs = torch.zeros_like(costs)
    # pbar = tqdm(range(len(problems)), total=len(problems))
    process_sequence = np.arange(len(problems))
    np.random.shuffle(process_sequence)
    pbar = tqdm(process_sequence, total=len(problems))
    for i in pbar:
        problem = problems[i]
        ref_image_id = problem.ref_image_id
        pbar.set_description(
            f"({i + 1}/{num_imgs}): " f"Run PatchMatch {ref_image_id:04} ...")
        depth, normal, cost = PatchMatcher.run_patch_match(i, verbose)
        save_dir = os.path.join(output_dir, f"{ref_image_id:04}")
        depth_path = os.path.join(
            save_dir,
            "depth_geom.npy",
        )
        normal_path = os.path.join(
            save_dir,
            "normal_geom.npy",
        )
        cost_path = os.path.join(
            save_dir,
            "cost_geom.npy",
        )
        np.save(depth_path, depth.cpu().numpy())
        np.save(normal_path, normal.cpu().numpy())
        np.save(cost_path, cost.cpu().numpy())

        output_depths[i] = depth
        output_normals[i] = normal
        output_costs[i] = cost

    return output_depths, output_normals, output_costs


def run_fusion(
    depths: torch.Tensor,
    normals: torch.Tensor,
    result_dir: str,
    problems: List[Problem],
    geom_consistent: int = 2,
    output_dir: Optional[str] = None,
    gi_iters=None,
) -> None:
    import gmvs.src as _C

    cpp_problems = []
    cameras = []

    num_imgs, height, width = depths.shape
    pbar = tqdm(range(num_imgs), total=num_imgs)

    print("Begin funsion")
    for i in pbar:
        pbar.set_description(
            f"({i + 1}/{num_imgs}): " f"Process image {problems[i].ref_image_id:04}"
        )
        problem = problems[i]
        ref_image_id = problem.ref_image_id
        camera_path = os.path.join(
            result_dir, "cams", f"{ref_image_id:04}_cam.txt")
        (K, R, t, (depth_min, depth_max)) = read_camera(camera_path)
        camera = _C.load_camera(K, R, t, height, width, depth_min, depth_max)
        cpp_problems.append(problem._to_cpp())
        cameras.append(camera)

    masks = torch.zeros([num_imgs, height, width]).bool().cuda()
    Fuser = _C.Fuser()
    Fuser.load_samples(
        cpp_problems,
        cameras,
        depths,
        normals,
        masks)

    if gi_iters is not None:
        os.makedirs(f"{output_dir}_gi{gi_iters}", exist_ok=True)

    valid = 0
    pbar = tqdm(range(num_imgs), total=num_imgs)
    for i in pbar:
        pbar.set_description(
            f"({i + 1}/{num_imgs}): " f"Fusing image and output depth/normal"
        )
        proj_depth, proj_normal = Fuser.run_fusion(i, geom_consistent)

        # Normalize the normal
        depth = proj_depth.cpu().numpy()
        normal = proj_normal.cpu().numpy()
        normal_norm = np.linalg.norm(normal, axis=-1, ord=2, keepdims=True)
        normal_norm = np.where(normal_norm > 0.5, normal_norm, 1.0)
        normal /= normal_norm
        valid += (depth > 0).sum()

        depth_noraml = np.concatenate(
            [depth[..., None], normal], axis=-1).astype(np.float32)
        problem = problems[i]
        ref_image_id = problem.ref_image_id
        if gi_iters is not None:
            save_path = os.path.join(
                f"{output_dir}_gi{gi_iters}", f"{(int(ref_image_id)):04d}.npy")
        else:
            save_path = os.path.join(
                f"{output_dir}", f"{(int(ref_image_id)):04d}.npy")
        np.save(save_path, depth_noraml)
    print(f"valid again: {valid}")
