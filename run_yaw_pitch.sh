#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHD_ROOT="$(cd "$REPO_DIR/../.." && pwd)"

# Example: yaw+pitch grid, still 1° yaw steps but discrete pitch values.
# Useful once yaw-only is stable.

python "$REPO_DIR/create_dataset.py" \
  --out_dir "$PHD_ROOT/data/active/phase_a_yaw_pitch" \
  --models_file "$REPO_DIR/models_phase_a_12.txt" \
  --mode rotate_object \
  --yaw_min -60 --yaw_max 60 --yaw_step 1 \
  --pitch_list -20 0 20 \
  --roll 0 \
  --dist 1.5 \
  --img_size 256
