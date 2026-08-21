#!/usr/bin/env bash
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PHD_ROOT="$(cd "$REPO_DIR/../.." && pwd)"

# Phase-A yaw-only dataset (10 models), fixed camera, yaw-only rotation.
# Matches the dist=0.20 smoke settings, but at 512x512.

conda run -n phd python "$REPO_DIR/create_dataset.py" \
  --out_dir "$PHD_ROOT/data/reference/phase_a_yaw_only_10obj_yaw2deg_512_nodof_dist0.20_yaw60_cam0.5_look0.1" \
  --models_file "$REPO_DIR/models_phase_a_10.txt" \
  --mode rotate_object \
  --yaw_min -60 --yaw_max 60 --yaw_step 2 \
  --pitch_list 0 --roll 0 \
  --dist 0.20 \
  --img_size 512 \
  --camera_height 0.5 \
  --look_at_height 0.1 \
  --test_frac 0.25
