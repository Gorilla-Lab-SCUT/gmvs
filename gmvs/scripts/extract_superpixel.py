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

    segment = felzenszwalb(resize_image, scale=100, sigma=0.5, min_size=5)  # for ScanNet

    n = 1
    for i in range(segment.max()):
        npixel = np.sum(segment == (i + 1))
        # if npixel > 0:
        if npixel > 384 * 288 / 128:  # ignore too small segs
            segment[segment == (i + 1)] = n
            n += 1
        else:
            segment[segment == (i + 1)] = 0

    segment = cv2.resize(segment, (w, h), interpolation=cv2.INTER_NEAREST)

    return segment


def seg_func(image_dir, save_dir, mask_dir = None, var_windlen: int = 15, var_thresh: float = 0.2, var_thresh2: float = -1):
    filename = os.path.basename(image_dir)
    rgb = cv2.imread(image_dir)
    gray = cv2.imread(image_dir, cv2.IMREAD_GRAYSCALE) / 255
    segment_ids = extract_superpixel(rgb)

    kernel = np.ones((11, 11), dtype=np.uint8)
    segment_close = cv2.morphologyEx(segment_ids / 255, cv2.MORPH_CLOSE, kernel)
    segment_ids = (segment_close * 255 + 0.5).astype(np.uint8)

    if mask_dir is not None:
        mask = np.load(mask_dir)  # single channel
        h, w = mask.shape
        segment_ids = np.where(mask==True, segment_ids, 0)

        all_seg_ids = np.unique(segment_ids)
        for segid in all_seg_ids[1:]: # jump 0
            area = np.where(segment_ids==segid)
            if len(area[0]) < h*w/128:
                segment_ids[area] = 0

    """ Efficient implementation """
    if var_thresh > 0.0 and var_windlen > 1:
        var_map = winVar(gray, wlen=var_windlen)
        var_map = var_map / var_map.max()
        var_map = var_map.astype(np.float32)
        segment_ids_map = np.where(var_map < var_thresh, segment_ids, 0).astype(np.int32)
    else:
        segment_ids_map = segment_ids

    """ Filter segs with variance map """
    if var_thresh2 > 0.0:
        from gmvs.lib.build import planar_initer as _C
        filter_segment_ids_map = _C.filter_by_var_map(var_map, segment_ids_map, var_thresh2)
    else:
        filter_segment_ids_map = segment_ids_map

    color_bar = np.random.randint(0, 255, [max(filter_segment_ids_map.max(), segment_ids.max()) + 1, 3])
    color_bar[0] = 0
    color_seg_var = color_bar[filter_segment_ids_map]
    color_seg = color_bar[segment_ids]
    rgb_segs = copy.deepcopy(rgb)
    rgb_segs[filter_segment_ids_map[..., None].repeat(3, axis=-1) == 0] = 0

    cv2.imwrite(
        os.path.join(save_dir, "segrgb", filename),
        np.concatenate((rgb, rgb_segs, color_seg_var, color_seg), axis=1).astype(np.uint8),
    )
    np.save(os.path.join(save_dir, "segid_npy", filename.replace(".png", ".npy")), segment_ids_map)


if __name__ == "__main__":
    args = get_args()
    image_dir = args.img_dir
    save_dir = args.save_dir

    os.makedirs(save_dir, exist_ok=True)
    os.makedirs(os.path.join(save_dir, "segrgb"), exist_ok=True)
    os.makedirs(os.path.join(save_dir, "segid_npy"), exist_ok=True)

    filenames = os.listdir(image_dir)
    filenames.sort()

    num_processes = 32
    assert num_processes > 0

    if num_processes == 1:
        """ Single preprocess """
        for filename in tqdm(filenames):
            image_path = os.path.join(image_dir, filename)
            seg_func(image_path, save_dir, None, 15, 0.2, -1)
    else:
        """ Multiple preprocess """
        queues = []
        for filename in tqdm(filenames):
            image_path = os.path.join(image_dir, filename)
            # queues.append((image_path, save_dir, None, 15, 0.2, -1))
            queues.append((image_path, save_dir, None, 15, -1, -1))  # for living_room
        start_process_pool(seg_func, queues, num_processes=32)
