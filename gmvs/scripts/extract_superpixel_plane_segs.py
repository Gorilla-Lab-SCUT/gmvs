# adapted from https://github.com/svip-lab/Indoor-SfMLearner/blob/master/extract_superpixel.py

import argparse
import copy
import multiprocessing
import os

import cv2
import numpy as np
from skimage.segmentation import felzenszwalb
from skimage.util import img_as_float
from tqdm import tqdm


def start_process_pool(worker_function, parameters, num_processes, timeout=None):
    if len(parameters) > 0:
        if num_processes <= 1:
            print(
                "Running loop for {} with {} calls on {} workers".format(
                    str(worker_function), len(parameters), num_processes
                )
            )
            results = []
            for c in parameters:
                results.append(worker_function(*c))
            return results
        print(
            "Running loop for {} with {} calls on {} subprocess workers".format(
                str(worker_function), len(parameters), num_processes
            )
        )
        with multiprocessing.Pool(processes=num_processes, maxtasksperchild=1) as pool:
            results = pool.starmap(worker_function, parameters)
            return results
    else:
        return None


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--img_dir", type=str, help="path to scannet data", required=True)
    parser.add_argument("--save_dir", type=str, help="path to scannet data", required=True)
    args = parser.parse_args()
    return args


def winVar(img, wlen: int = 3):
    """calculate the variance map of an image"""
    wmean, wsqrmean = (
        cv2.boxFilter(x, -1, (wlen, wlen), borderType=cv2.BORDER_REFLECT) for x in (img, img * img)
    )
    return np.sqrt(np.abs(wsqrmean - wmean * wmean))


def extract_superpixel(image, weight=None) -> np.ndarray:
    image = image * weight if weight is not None else image
    h, w, c = image.shape

    resize_image = cv2.resize(image, (384, 288))
    resize_image = img_as_float(resize_image)

    # segment = felzenszwalb(resize_image, scale=100, sigma=0.5, min_size=50)
    segment = felzenszwalb(resize_image, scale=100, sigma=0.5, min_size=5)

    n = 1
    for i in range(segment.max()):
        npixel = np.sum(segment == (i + 1))
        if npixel > 384 * 288 / 128:  # ignore too small segs
            segment[segment == (i + 1)] = n
            n += 1
        else:
            segment[segment == (i + 1)] = 0

    segment = cv2.resize(segment, (w, h), interpolation=cv2.INTER_NEAREST)

    return segment


def seg_func(image_dir, save_dir, var_windlen: int = 15, var_thesh: float = 0.2):
    filename = os.path.basename(image_dir)
    rgb = cv2.imread(image_dir)
    H, W, _ = rgb.shape
    gray = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE) / 255
    segment = extract_superpixel(rgb)

    kernel = np.ones((11, 11), dtype=np.uint8)
    segment_close = cv2.morphologyEx(segment / 255, cv2.MORPH_CLOSE, kernel)
    segment = (segment_close * 255 + 0.5).astype(np.uint8)

    """ Efficient implementation """
    img = winVar(gray, wlen=var_windlen)
    img = img / img.max()
    segment_var = np.where(img < var_thesh, segment, 0)

    plane_segs = cv2.imread(
        os.path.join(os.path.dirname(image_dir), "../semantic_deeplab", filename)
    )
    # TODO: remove small pieces
    plane_mask = plane_segs > 0

    keep_segment_var = copy.deepcopy(segment_var)
    segment_var_plane = segment_var * plane_mask[..., 0]
    seg_ids = np.unique(segment_var)[1:]
    max_seg_ids = seg_ids.max()
    assert max_seg_ids < 128
    plane_seg_bias = 128
    for segid in seg_ids:
        area_ = (segment_var == segid).sum()
        area_plane = (segment_var_plane == segid).sum()
        if area_plane / area_ > 0.5:  # both plane and non-plane in one segs
            keep_segment_var[segment_var_plane == segid] = (
                segid + plane_seg_bias
            )  # relabel the plane region

    # filter too small area in new segs
    seg_ids = np.unique(keep_segment_var)[1:]
    for segid in seg_ids:
        area_keep = (keep_segment_var == segid).sum()
        if area_keep < H * W / 128:
            keep_segment_var[keep_segment_var == segid] = 0

    color_bar = np.random.randint(0, 255, [max(keep_segment_var.max(), segment_var.max()) + 1, 3])
    color_bar[0] = 0
    color_seg_var_plane = color_bar[keep_segment_var]
    color_seg_var_plane_white = copy.deepcopy(color_seg_var_plane)
    color_seg_var_plane_white[keep_segment_var > plane_seg_bias] = 255
    rgb_plane = copy.deepcopy(rgb)
    rgb_plane[keep_segment_var[..., None].repeat(3, axis=-1) < plane_seg_bias] = 0
    color_seg_var = color_bar[segment_var]

    cv2.imwrite(
        os.path.join(save_dir, "segrgb", filename),
        np.concatenate(
            (
                np.concatenate((rgb, plane_mask * 254), axis=0),
                np.concatenate((color_seg_var, color_seg_var_plane_white), axis=0),
                np.concatenate((color_seg_var_plane, rgb_plane), axis=0),
            ),
            axis=1,
        ).astype(np.uint8),
    )
    np.save(os.path.join(save_dir, "segid_npy", filename.replace(".png", ".npy")), keep_segment_var)


if __name__ == "__main__":
    args = get_args()
    image_dir = args.img_dir
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "segrgb"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "segid_npy"), exist_ok=True)

    filenames = os.listdir(image_dir)
    filenames.sort()

    queues = []
    for filename in tqdm(filenames):
        image_path = os.path.join(image_dir, filename)
        queues.append((image_path, save_dir, 15, 0.2))
    start_process_pool(seg_func, queues, num_processes=32)

    # for filename in tqdm(filenames):
    #     rgb = cv2.imread(os.path.join(image_dir, filename))
    #     H, W, _ = rgb.shape
    #     gray = cv2.imread(os.path.join(image_dir, filename), cv2.IMREAD_GRAYSCALE) / 255
    #     segment = extract_superpixel(rgb)

    #     kernel = np.ones((11, 11), dtype=np.uint8)
    #     segment_close = cv2.morphologyEx(segment/255, cv2.MORPH_CLOSE, kernel)
    #     segment = (segment_close*255+0.5).astype(np.uint8)

    #     # #####################################
    #     # # debug code
    #     # #####################################
    #     # color_bar = np.random.randint(0, 255, [segment.max() + 1, 3])
    #     # color_bar[0] = 0
    #     # color_seg= color_bar[segment]

    #     # cv2.imwrite(os.path.join(save_dir, "segrgb", filename),
    #     #             np.concatenate((rgb, color_seg), axis=0).astype(np.uint8))

    #     # #####################################

    #     ''' Efficient implementation '''
    #     img = winVar(gray, wlen=15)
    #     img = img / img.max()

    #     segment_var = np.where(img < 0.2, segment, 0)

    #     plane_segs = cv2.imread(os.path.join(image_dir, "../semantic_deeplab", filename))
    #     # TODO: remove small pieces
    #     plane_mask = plane_segs > 0

    #     keep_segment_var = copy.deepcopy(segment_var)
    #     segment_var_plane = segment_var*plane_mask[..., 0]
    #     seg_ids = np.unique(segment_var)[1:]
    #     max_seg_ids = seg_ids.max()
    #     assert max_seg_ids < 128
    #     plane_seg_bias = 128
    #     for segid in seg_ids:
    #         area_ = (segment_var == segid).sum()
    #         area_plane = (segment_var_plane == segid).sum()
    #         if area_plane / area_ > 0.5:  # both plane and non-plane in one segs
    #             keep_segment_var[segment_var_plane == segid] = segid + plane_seg_bias  # relabel the plane region

    #     # filter too small area in new segs
    #     seg_ids = np.unique(keep_segment_var)[1:]
    #     for segid in seg_ids:
    #         area_keep = (keep_segment_var == segid).sum()
    #         if area_keep < H*W / 128:
    #             keep_segment_var[keep_segment_var == segid] = 0

    #     color_bar = np.random.randint(0, 255, [max(keep_segment_var.max(), segment_var.max()) + 1, 3])
    #     color_bar[0] = 0
    #     color_seg_var_plane = color_bar[keep_segment_var]
    #     color_seg_var_plane_white = copy.deepcopy(color_seg_var_plane)
    #     color_seg_var_plane_white[keep_segment_var > plane_seg_bias] = 255
    #     rgb_plane = copy.deepcopy(rgb)
    #     rgb_plane[keep_segment_var[..., None].repeat(3, axis=-1)<plane_seg_bias] = 0
    #     color_seg_var = color_bar[segment_var]

    #     cv2.imwrite(os.path.join(save_dir, "segrgb", filename),
    #         np.concatenate((
    #             np.concatenate((rgb, plane_mask*254), axis=0),
    #             np.concatenate((color_seg_var, color_seg_var_plane_white), axis=0),
    #             np.concatenate((color_seg_var_plane, rgb_plane), axis=0)),
    #         axis=1).astype(np.uint8))
    #     np.save(os.path.join(save_dir, "segid_npy", filename.replace(".png", ".npy")), keep_segment_var)
