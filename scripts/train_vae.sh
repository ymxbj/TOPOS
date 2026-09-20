#!/usr/bin/env bash
# Single node, multi-GPU VAE training.
# For single-GPU, use: python train_vae.py --config configs/vae_train.yaml
torchrun --nproc_per_node=auto train_vae.py --config configs/vae_train.yaml "$@"
