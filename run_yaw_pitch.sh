#!/usr/bin/env bash
set -euo pipefail

# Example: yaw+pitch grid, still 1° yaw steps but discrete pitch values.
# Useful once yaw-only is stable.

python create_dataset.py \
  --out_dir ./phase_a_yaw_pitch \
  --models_file ./models_4.txt \
  --mode rotate_object \
  --yaw_min -60 --yaw_max 60 --yaw_step 1 \
  --pitch_list -20 0 20 \
  --roll 0 \
  --dist 1.5 \
  --img_size 256
