# gorilla-mvs

## Installing
```sh
git submodule update --init --recursive
python setup.py develop
```
> Installation is so ugly, because it requires `torch` and `opencv`, but the `pybind11` in `torch` can not handle the `opencv` lib.
> So we split the extension sources into two parts and use two install scripts.
> 
> You can also refer to the [torch-mvs](https://github.com/lzhnb/torch_mvs) (without `opencv`), but we did align their results.

## Preprocessing
```sh
python -m gmvs.scripts.colmap2mvs --dense_folder $DATA_DIR/dense/ --save_folder $RESULT_ROOT
```

## Extract superpixels
```sh
python -m gmvs.scripts.extract_superpixel --img_dir $DATA_DIR/images --save_dir $SUPERPIXEL_DIR
```

## Running
```sh
python -m gmvs.scripts.launch -rd $RESULT_ROOT # --mvs_suffix $MVS_SUFFIX)(optional)
```
If with depth and normal input as initialization
```sh
python -m gmvs.scripts.launch -rd $RESULT_ROOT --mvs_suffix $MVS_SUFFIX --dn_input --input_depth_normal_dir $INPUT_DEPTH_NORMAL_DIR
```

## Postprocessing
```sh
'''runing on cpu'''
python -m gmvs.scripts.mvs_fusion_segmentaion_cpu --depth_normal_dir $MVS_DIR/depth_normal/ \
        --data_dir $DATA_DIR --superpixel_dir $SUPERPIXEL_DIR/ \
        --save_dir $MVS_DIR/planar_prior/ --vis --clean_mesh # (--gen_mask --mask_dir planar_mask_mvs_clean) for init 
'''runing on cuda'''
python -m gmvs.scripts.mvs_fusion_segmentaion --depth_normal_dir $MVS_DIR/depth_normal/ \
        --data_dir $DATA_DIR --superpixel_dir $SUPERPIXEL_DIR/ \
        --save_dir $MVS_DIR/planar_prior/ --vis --clean_mesh # (--gen_mask --mask_dir planar_mask_mvs_clean) for init 
```

