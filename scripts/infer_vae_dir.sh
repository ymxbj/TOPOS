#!/usr/bin/env bash
# Reconstruct meshes for every .npz under a directory. Override the directory
# on the CLI, e.g.
#   bash scripts/infer_vae_dir.sh DATA.test.test_pointcloud_dir ./my_pointclouds
python3 train_vae.py --config configs/vae_test_dir.yaml "$@"
