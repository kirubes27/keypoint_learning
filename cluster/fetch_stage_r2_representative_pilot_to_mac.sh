#!/bin/bash
# Run on the Mac after the representative pilot has finished.

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Run this collector on the Mac after leaving SSH with: exit" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
MAC_PHD_ROOT="$(cd "$REPO_ROOT/.." && pwd)"
REMOTE_HOST="${REMOTE_HOST:-ko75kamy@lcluster13.hrz.tu-darmstadt.de}"
REMOTE_REPO="${REMOTE_REPO:-/work/scratch/ko75kamy/keypoint_learning}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_lichtenberg}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$MAC_PHD_ROOT/cluster_downloads}"
REMOTE_OUTPUT="keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r2_representative_pilot"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$DOWNLOAD_ROOT/stage_r2_representative_pilot_$STAMP"
ARCHIVE="$DEST/stage_r2_representative_pilot.tgz"

[[ -f "$SSH_KEY" ]] || { echo "Missing SSH key: $SSH_KEY" >&2; exit 2; }
mkdir -p "$DEST"

ssh -i "$SSH_KEY" "$REMOTE_HOST" bash -s -- "$REMOTE_REPO" "$REMOTE_OUTPUT" \
  > "$ARCHIVE" <<'REMOTE_SCRIPT'
set -euo pipefail
repo="$1"
output="$2"
cd "$repo"
run="$output/runs/coordinate_standard64_k10_seed41"
required=(
  "$run/config.json"
  "$run/training_summary.json"
  "$run/best_validation_metrics.json"
  "$run/best_coordinate_path_probe.json"
  "$run/coordinate_path_probe_history.json"
  "$output/REPRESENTATIVE_PILOT_SUMMARY.json"
)
for path in "${required[@]}"; do
  [[ -s "$path" ]] || { echo "Pilot incomplete: missing $path" >&2; exit 3; }
done
mapfile -t logs < <(find slurm_logs -maxdepth 1 -type f -name 'stage_r2_pilot_*' -print)
echo "Validated pilot; downloading result directory and ${#logs[@]} log files." >&2
tar -czf - "$output" "${logs[@]}"
REMOTE_SCRIPT

tar -xzf "$ARCHIVE" -C "$DEST"
shasum -a 256 "$ARCHIVE" > "$ARCHIVE.sha256"
SUMMARY="$DEST/$REMOTE_OUTPUT/REPRESENTATIVE_PILOT_SUMMARY.json"
[[ -s "$SUMMARY" ]] || { echo "Downloaded pilot has no summary." >&2; exit 4; }
printf '%s\n' "$DEST" > "$DOWNLOAD_ROOT/STAGE_R2_REPRESENTATIVE_PILOT_LATEST.txt"
echo "Saved and validated on the Mac."
echo "Folder: $DEST"
echo "Summary: $SUMMARY"
