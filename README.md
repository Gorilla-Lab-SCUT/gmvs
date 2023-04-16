# gorilla-mvs

## Installing
```sh
git submodule update --init --recursive
python setup.py develop
```

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

<!-- 
      baseline / gi1_gc2 / gi2_gc3
0050: 4377620  / 7352799 / 4833486
0084: 1748901  / 2969206 / 1776719
0580: 5646921  / 8995206 / 6153378
0616: 1436371  / 2259463 / 1465421 
-->
