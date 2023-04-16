# Copyright (c) Gorilla-Lab. All rights reserved.
import os
import shutil
from typing import List, Optional, Tuple, Type

import cv2
import numpy as np

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


def initialization(
    result_dir: str,
    output_dir: str,
    problem: Problem,
    verbose: bool = False,
    init_plane: bool = False,
    no_init: bool = False,
    superpixel_dir: Optional[str] = None,
    input_depth_normal_dir: Optional[str] = None,
) -> None:
    # read the id of the reference image
    ref_image_id = problem.ref_image_id

    # define the save_dir
    save_dir = os.path.join(output_dir, f"{ref_image_id:04}")
    os.makedirs(save_dir, exist_ok=True)

    if init_plane:
        # read the depth and normal files
        depth_normal_dir = os.path.join(input_depth_normal_dir, f"{ref_image_id:04}")
        shutil.copy2(
            os.path.join(depth_normal_dir, "depth.npy"), os.path.join(save_dir, "depth.npy")
        )
        shutil.copy2(
            os.path.join(depth_normal_dir, "normal.npy"), os.path.join(save_dir, "normal.npy")
        )
        shutil.copy2(os.path.join(depth_normal_dir, "cost.npy"), os.path.join(save_dir, "cost.npy"))
    elif no_init:
        from gmvs.lib.build import planar_initer as _C

        initer = _C.PlanarInit(verbose)

        # get the folder storing images and camera parameters
        image_dir = os.path.join(result_dir, "images")
        cam_dir = os.path.join(result_dir, "cams")

        # get the filepath of the ref_image and read as a gray image
        image_path = os.path.join(image_dir, f"{ref_image_id:04}.jpg")
        initer.add_image(image_path)
        # read camear
        height, width = initer.get_image(0).shape
        cam_path = os.path.join(cam_dir, f"{ref_image_id:04}_cam.txt")
        (K, R, t, (depth_min, depth_max)) = read_camera(cam_path)
        initer.add_camera(K, R, t, height, width, depth_min, depth_max)

        # get the src_images for the ref_image
        num_src_images = len(problem.src_image_ids)
        for i in range(num_src_images):
            src_image_id = problem.src_image_ids[i]
            # load image and camera
            image_path = os.path.join(image_dir, f"{src_image_id:04}.jpg")
            initer.add_image(image_path)
            cam_path = os.path.join(cam_dir, f"{src_image_id:04}_cam.txt")
            (K, R, t, (depth_min, depth_max)) = read_camera(cam_path)
            initer.add_camera(K, R, t, height, width, depth_min, depth_max)

        # get the minimum and maximum of depth according to the ref_image's
        initer.params.depth_min = initer.get_camera(0).depth_min * 0.6
        initer.params.depth_max = initer.get_camera(0).depth_max * 1.2
        if verbose:
            print(f"depth range: {initer.params.depth_min} {initer.params.depth_max}")

        # get the number of images to process(1 + num_src)
        initer.params.num_images = initer.get_num_images()
        if verbose:
            print(f"num images: {initer.params.num_images}")

        # init the depth/normals/costs
        # cuda space init for parallel
        initer.cuda_space_init()

        # parallel patch match
        initer.run_patch_match()

        depth = np.zeros([height, width], dtype=np.float32)
        normal = np.zeros([height, width, 3], dtype=np.float32)
        cost = np.zeros([height, width], dtype=np.float32)
        initer.export_plane(depth, normal, cost)

        # save results
        depth_path = os.path.join(save_dir, "depth.npy")
        np.save(depth_path, depth)
        normal_path = os.path.join(save_dir, "normal.npy")
        np.save(normal_path, normal)
        cost_path = os.path.join(save_dir, "cost.npy")
        np.save(cost_path, cost)

    else:
        from gmvs.lib.build import planar_initer as _C

        initer = _C.PlanarInit(verbose)

        # get the folder storing images and camera parameters
        image_dir = os.path.join(result_dir, "images")
        cam_dir = os.path.join(result_dir, "cams")

        # get the filepath of the ref_image and read as a gray image
        image_path = os.path.join(image_dir, f"{ref_image_id:04}.jpg")
        initer.add_image(image_path)
        # read camear
        height, width = initer.get_image(0).shape
        cam_path = os.path.join(cam_dir, f"{ref_image_id:04}_cam.txt")
        (K, R, t, (depth_min, depth_max)) = read_camera(cam_path)
        initer.add_camera(K, R, t, height, width, depth_min, depth_max)

        # get the src_images for the ref_image
        num_src_images = len(problem.src_image_ids)
        for i in range(num_src_images):
            src_image_id = problem.src_image_ids[i]
            # load image and camera
            image_path = os.path.join(image_dir, f"{src_image_id:04}.jpg")
            initer.add_image(image_path)
            cam_path = os.path.join(cam_dir, f"{src_image_id:04}_cam.txt")
            (K, R, t, (depth_min, depth_max)) = read_camera(cam_path)
            initer.add_camera(K, R, t, height, width, depth_min, depth_max)

        # add superpixel
        if superpixel_dir:
            superpixel_path = os.path.join(superpixel_dir, f"{ref_image_id:04d}.npy")
            superpixel = np.load(superpixel_path)
            initer.add_superpixel(superpixel)

        # get the minimum and maximum of depth according to the ref_image's
        initer.params.depth_min = initer.get_camera(0).depth_min * 0.6
        initer.params.depth_max = initer.get_camera(0).depth_max * 1.2
        if verbose:
            print(f"depth range: {initer.params.depth_min} {initer.params.depth_max}")

        # get the number of images to process(1 + num_src)
        initer.params.num_images = initer.get_num_images()
        if verbose:
            print(f"num images: {initer.params.num_images}")

        # init the depth/normals/costs
        # cuda space init for parallel
        initer.cuda_space_init()

        # parallel patch match
        initer.run_patch_match()

        depth = np.zeros([height, width], dtype=np.float32)
        normal = np.zeros([height, width, 3], dtype=np.float32)
        cost = np.zeros([height, width], dtype=np.float32)
        initer.export_plane(depth, normal, cost)

        if verbose:
            print("Run Planar Prior Assisted PatchMatch MVS ...")
        initer.params.planar_prior = True
        initer.params.superpixel_filter = superpixel_dir is not None
        # triangulation and run patch match again
        triangulation_path = os.path.join(
            save_dir, "triangulation.png"
        )  # TODO : do not save it in release version
        initer.triangulation(depth, triangulation_path)
        initer.cuda_planar_prior_init()
        initer.run_patch_match()
        initer.export_plane(depth, normal, cost)

        # save results
        depth_path = os.path.join(save_dir, "depth.npy")
        np.save(depth_path, depth)
        normal_path = os.path.join(save_dir, "normal.npy")
        np.save(normal_path, normal)
        cost_path = os.path.join(save_dir, "cost.npy")
        np.save(cost_path, cost)
