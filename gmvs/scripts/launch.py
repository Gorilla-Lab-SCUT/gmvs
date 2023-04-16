# Copyright (c) Gorilla-Lab. All rights reserved.
import argparse
import os
import random
from typing import List

import numpy as np
from tqdm import tqdm

import gmvs
from gmvs.consistency import run_fusion, run_patch_match
from gmvs.planar_init import initialization

seed = 3407
random.seed(seed)
np.random.seed(seed)
os.environ["PYTHONHASHSEED"] = str(seed)

listfiles = lambda root: [f for base, _, files in os.walk(root) if files for f in files]


def generate_sample_list(result_dir: str) -> List[gmvs.Problem]:
    pair_file = os.path.join(result_dir, "pair.txt")
    assert os.path.exists(pair_file), f"{pair_file} does not exist"

    # parse pair.txt
    with open(pair_file, "r") as f:
        content = f.read()
    lines = content.split("\n")

    problems: List[gmvs.Problem] = []
    num_images = int(lines[0])
    for i in range(num_images):
        # read reference id
        ref_id = int(lines[i * 2 + 1])
        # read source ids
        src_ids = []
        src_line = lines[i * 2 + 2].split(" ")
        num_src_images = int(src_line[0])
        for j in range(num_src_images):
            src_id = int(src_line[j * 2 + 1])
            score = float(src_line[j * 2 + 2])
            if score < 0.0:
                continue
            src_ids.append(src_id)

        problem = gmvs.Problem(ref_id, src_ids)
        problems.append(problem)

    return problems


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--res_dir",
        "-rd",
        type=str,
        required=True,
        help="root directory to store the results processed by the colmap2mvsnet_cam.py",
    )
    parser.add_argument(
        "--mvs_suffix",
        "-ms",
        type=str,
        default="mvs",
        help="sub-directory under the root to store the mvs results output by this script",
    )
    parser.add_argument(
        "--input_depth_normal_dir",
        "-idnd",
        type=str,
        default=None,
        help="directory to store the input depth and normal for initialization",
    )
    parser.add_argument(
        "--dn_input",
        "-di",
        action="store_true",
        help="if use depth and normal input for initialization, please make sure there is */depth.npy and */normal.npy under input_depth_normal_dir",
    )
    parser.add_argument(
        "--superpixel_dir",
        "-sd",
        type=str,
        default=None,
        help="directory to store the superpixel numpy array",
    )
    """ default parames below here"""
    parser.add_argument(
        "--geom_iters",
        "-gi",
        type=int,
        default=2,
        help="number of geometric constraint iterations",
    )
    parser.add_argument(
        "--geom_cons",
        "-gc",
        type=int,
        default=3,
        help="number of geometric consistent to filter the fusion point cloud, default to 2",
    )
    parser.add_argument(
        "--no_init",
        "-ni",
        action="store_true",
        help="not initialization",
    )
    parser.add_argument(
        "--verbose",
        "-v",
        action="store_true",
        help="verbose description",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = get_args()

    import time
    time_tic = time.time()

    result_root = args.res_dir

    problems = generate_sample_list(result_root)
    num_imgs = len(problems)
    print(f"there are {num_imgs} problems to be processed!")

    # init output dir
    output_dir = os.path.join(result_root, args.mvs_suffix)
    os.makedirs(output_dir, exist_ok=True)
    output_depth_normal_dir = os.path.join(output_dir, "depth_normal")
    os.makedirs(output_depth_normal_dir, exist_ok=True)

    # init for mvs
    pbar = tqdm(range(num_imgs), total=num_imgs)
    for i in pbar:
        pbar.set_description(
            f"({i + 1}/{num_imgs}): " f"Init image {problems[i].ref_image_id:04} ..."
        )
        initialization(
            result_dir=result_root,
            output_dir=output_dir,
            problem=problems[i],
            verbose=args.verbose,
            init_plane=args.dn_input,
            no_init=args.no_init,
            superpixel_dir=args.superpixel_dir,
            input_depth_normal_dir=args.input_depth_normal_dir,
        )

    for i in range(args.geom_iters):
        depths, normals, costs = run_patch_match(
            result_dir=result_root,
            output_dir=output_dir,
            problems=problems,
            verbose=args.verbose,
            first_stage=(i == 0),
        )

    if args.geom_iters > 0:
        run_fusion(
            depths,
            normals,
            result_root,
            problems,
            geom_consistent=int(args.geom_cons),
            output_dir=output_depth_normal_dir,
        )

    time_toc = time.time() - time_tic
    print("Time Used for MVS: ", time_toc)
