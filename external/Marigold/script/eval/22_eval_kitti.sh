#!/usr/bin/env bash
set -e
set -x


python eval.py \
    --base_data_dir $BASE_DATA_DIR \
    --dataset_config config/dataset/data_kitti_eigen_test.yaml \
    --alignment least_square \
    --prediction_dir output/kitti_eigen_test/prediction \
    --output_dir output/kitti_eigen_test/eval_metric \
