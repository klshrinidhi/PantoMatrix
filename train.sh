#! /bin/bash

set -o nounset
set -o errexit
set -o xtrace

export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=3
# python \
#     scripts/EMAGE_2024/train.py \
#     --config scripts/EMAGE_2024/configs/cnn_vqvae_lower_30.yaml
# python \
#     scripts/EMAGE_2024/train.py \
#     --config scripts/EMAGE_2024/configs/cnn_vqvae_upper_30.yaml
python \
    scripts/EMAGE_2024/train.py \
    --config scripts/EMAGE_2024/configs/cnn_vqvae_hands_30.yaml
# python \
#     scripts/EMAGE_2024/train.py \
#     --config scripts/EMAGE_2024/configs/cnn_vqvae_full_30.yaml
# python \
#     scripts/EMAGE_2024/train.py \
#     --config scripts/EMAGE_2024/configs/cnn_vqvae_face_30.yaml
# python \
#     scripts/EMAGE_2024/train.py \
#     --config scripts/EMAGE_2024/configs/cnn_vqvae_all_30.yaml
