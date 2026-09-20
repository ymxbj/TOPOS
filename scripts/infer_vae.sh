#!/usr/bin/env bash
# Reconstruct one mesh from a single point cloud (.npz with `points`, `normals`).
# Override the input on the CLI, e.g.
#   bash scripts/infer_vae.sh DATA.test.test_pointcloud_path ./examples/face.npz
python3 train_vae.py --config configs/vae_test.yaml "$@"
