#!/usr/bin/env bash
# Single node, multi-GPU DiT training.
torchrun --nproc_per_node=auto train_dit.py --config configs/dit_train.yaml "$@"
