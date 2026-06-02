#!/usr/bin/env bash
set -euo pipefail

# Sanity render by default. Add --full to render the complete 0..358 step-2 dataset.
conda run -n phd python generate_tdw_world_z_roll_dataset.py \
  --out_dir ./_tdw_world_z_roll_base_panel_512_v2 \
  --models_file ./models_affine_final_6.txt \
  --img_size 512 \
  --camera_dist 0.56 \
  --object_base_y 1.0 \
  --camera_height 0.0 \
  --look_at_height 0.0 \
  --theta_step_deg 2 \
  "$@"
