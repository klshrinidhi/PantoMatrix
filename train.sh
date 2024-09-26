#! /bin/bash

set -o nounset
set -o errexit
set -o xtrace

export CUDA_VISIBLE_DEVICES=0
python \
    scripts/EMAGE_2024/train.py \
    --config scripts/EMAGE_2024/configs/cnn_vqvae_upper_30.yaml
