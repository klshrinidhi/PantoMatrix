#! /bin/bash

set -o nounset
set -o errexit
set -o xtrace

export CUDA_VISIBLE_DEVICES=4
python \
    scripts/EMAGE_2024/test.py \
    --config scripts/EMAGE_2024/configs/cnn_vqvae_upper_30.yaml
