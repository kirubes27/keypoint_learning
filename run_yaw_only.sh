#!/usr/bin/env bash
set -euo pipefail

# Example: yaw-only, 1° steps, fixed pitch=0, fixed distance.
# Edit paths + model list as needed.

python create_dataset.py \
  --out_dir ./phase_a_yaw_only \
  --models_file ./models_4.txt \
  --mode rotate_object \
  --yaw_min -90 --yaw_max 90 --yaw_step 1 \
  --pitch_list 0 \
  --roll 0 \
  --dist 1.5 \
  --img_size 256
