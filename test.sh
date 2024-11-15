#! /bin/bash

set -o nounset
set -o errexit
set -o xtrace

export OMP_NUM_THREADS=2
export CUDA_VISIBLE_DEVICES=7
rm -rf meshes

rm -rf datasets/cache/lower_a30b2/test/ || true
rm -rf datasets/cache/upper_a30b2/test/ || true
rm -rf datasets/cache/hands_a30b2/test/ || true

# python \
#     scripts/EMAGE_2024/test.py \
#     --stat ts \
#     --config scripts/EMAGE_2024/configs/cnn_vqvae_lower_30_dsus.yaml
# python \
#     scripts/EMAGE_2024/test.py \
#     --stat ts \
#     --config scripts/EMAGE_2024/configs/cnn_vqvae_upper_30_dsus.yaml
# python \
#     scripts/EMAGE_2024/test.py \
#     --stat ts \
#     --config scripts/EMAGE_2024/configs/cnn_vqvae_hands_30_dsus.yaml

python \
    scripts/EMAGE_2024/test.py \
    --stat ts \
    --config scripts/EMAGE_2024/configs/cnn_vqvae_lower_30.yaml
python \
    scripts/EMAGE_2024/test.py \
    --stat ts \
    --config scripts/EMAGE_2024/configs/cnn_vqvae_upper_30.yaml
python \
    scripts/EMAGE_2024/test.py \
    --stat ts \
    --config scripts/EMAGE_2024/configs/cnn_vqvae_hands_30.yaml

# python \
#     scripts/EMAGE_2024/test.py \
#     --config scripts/EMAGE_2024/configs/cnn_vqvae_face_30.yaml

# python \
#     scripts/EMAGE_2024/test.py \
#     --config scripts/EMAGE_2024/configs/cnn_vqvae_full_30.yaml

# python \
#     scripts/EMAGE_2024/test.py \
#     --config scripts/EMAGE_2024/configs/cnn_vqvae_all_30.yaml
