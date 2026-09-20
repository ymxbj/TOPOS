#!/usr/bin/env bash
# Batch inference over a directory of images. Override DATA.test.test_image_dir to point at your images.
python3 train_dit.py --config configs/dit_test_dir.yaml "$@"
