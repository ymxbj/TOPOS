#!/usr/bin/env bash
# Single image -> single mesh. Override DATA.test.test_image_path or DATA.test.seeds on the CLI, e.g.
#   bash scripts/infer_image.sh DATA.test.test_image_path ./examples/face.png DATA.test.seeds "[0,1,2]"
python3 train_dit.py --config configs/dit_test.yaml "$@"
