#!/bin/bash
# Run on the Mac after the three Stage-R1 cluster tasks have finished.

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "Run this collector on the Mac after leaving SSH with: exit" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GIT_COMMON_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir)"
MAC_PHD_ROOT="${MAC_PHD_ROOT:-$(cd "$GIT_COMMON_DIR/../../.." && pwd)}"
REMOTE_HOST="${REMOTE_HOST:-ko75kamy@lcluster13.hrz.tu-darmstadt.de}"
REMOTE_REPO="${REMOTE_REPO:-/work/scratch/ko75kamy/keypoint_learning}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_lichtenberg}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$MAC_PHD_ROOT/artifacts/experiments/2026-07-05-stage-r1-shape-gate/cluster-results}"
REMOTE_OUTPUT="keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_shape_gate"
STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$DOWNLOAD_ROOT/stage_r1_shape_gate_$STAMP"
ARCHIVE="$DEST/stage_r1_shape_gate.tgz"

[[ -f "$SSH_KEY" ]] || { echo "Missing SSH key: $SSH_KEY" >&2; exit 2; }
mkdir -p "$DEST"

ssh -i "$SSH_KEY" "$REMOTE_HOST" bash -s -- "$REMOTE_REPO" "$REMOTE_OUTPUT" \
  > "$ARCHIVE" <<'REMOTE_SCRIPT'
set -euo pipefail
repo="$1"
output="$2"
cd "$repo"
count="$(find "$output/tiny_overfit" -name metrics.json -type f | wc -l)"
if [[ "$count" -ne 3 ]]; then
  echo "Stage R1 is incomplete: found $count/3 metrics.json files." >&2
  exit 3
fi
mapfile -t logs < <(find slurm_logs -maxdepth 1 -type f -name 'stage_r1_shape_*' -print)
echo "Validated 3/3 runs; downloading results and ${#logs[@]} log files." >&2
tar -czf - "$output" "${logs[@]}"
REMOTE_SCRIPT

tar -xzf "$ARCHIVE" -C "$DEST"
shasum -a 256 "$ARCHIVE" > "$ARCHIVE.sha256"
R1_ROOT="$DEST/$REMOTE_OUTPUT"
python "$REPO_ROOT/keypoint_net/diagnostics/summarize_stage_r1_shape_gate.py" \
  --root "$R1_ROOT" > "$DEST/summary_console.json"

SUMMARY="$R1_ROOT/R1_SHAPE_GATE_SUMMARY.json"
CSV="$R1_ROOT/R1_SHAPE_GATE_RUNS.csv"
[[ -s "$SUMMARY" && -s "$CSV" ]] || {
  echo "Downloaded runs, but local R1 summaries were not created." >&2
  exit 4
}

printf '%s\n' "$DEST" > "$DOWNLOAD_ROOT/STAGE_R1_SHAPE_GATE_LATEST.txt"
echo "Saved and summarized on the Mac."
echo "Folder: $DEST"
echo "Summary: $SUMMARY"
echo "Runs table: $CSV"
