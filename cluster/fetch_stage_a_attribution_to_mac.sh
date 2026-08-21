#!/bin/bash
# Run this script on the Mac. It validates the six cluster runs, downloads all
# result files and matching scheduler logs, and creates the local JSON/CSV
# summary. It deliberately keeps terminal output short.

set -euo pipefail

if [[ "$(uname -s)" != "Darwin" ]]; then
  echo "This collector must be run on the Mac after leaving the cluster with: exit" >&2
  exit 2
fi

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
GIT_COMMON_DIR="$(git -C "$REPO_ROOT" rev-parse --path-format=absolute --git-common-dir)"
MAC_PHD_ROOT="${MAC_PHD_ROOT:-$(cd "$GIT_COMMON_DIR/../../.." && pwd)}"

REMOTE_HOST="${REMOTE_HOST:-ko75kamy@lcluster13.hrz.tu-darmstadt.de}"
REMOTE_REPO="${REMOTE_REPO:-/work/scratch/ko75kamy/keypoint_learning}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_lichtenberg}"
DOWNLOAD_ROOT="${DOWNLOAD_ROOT:-$MAC_PHD_ROOT/artifacts/experiments/2026-07-05-stage-a-attribution/cluster-results}"
BASELINE_ROOT="${BASELINE_ROOT:-$MAC_PHD_ROOT/artifacts/experiments/2026-07-05-stage-a-k-sweep/cluster-results/stage_a_k_sweep_53032295/keypoint_net/diagnostics/outputs/final_material_keypoints/stage_a_k_sweep}"

STAMP="$(date +%Y%m%d_%H%M%S)"
DEST="$DOWNLOAD_ROOT/stage_a_attribution_$STAMP"
ARCHIVE="$DEST/stage_a_attribution.tgz"
REMOTE_OUTPUT="keypoint_net/diagnostics/outputs/final_material_keypoints/stage_a_attribution"

if [[ ! -f "$SSH_KEY" ]]; then
  echo "Missing SSH key: $SSH_KEY" >&2
  exit 2
fi
if [[ ! -d "$BASELINE_ROOT" ]]; then
  echo "Missing local K=10 baseline results: $BASELINE_ROOT" >&2
  exit 2
fi

mkdir -p "$DEST"

# The remote command emits only a gzip archive on stdout. Status information is
# sent to stderr, preventing it from corrupting the downloaded archive.
ssh -i "$SSH_KEY" "$REMOTE_HOST" bash -s -- "$REMOTE_REPO" "$REMOTE_OUTPUT" \
  > "$ARCHIVE" <<'REMOTE_SCRIPT'
set -euo pipefail
repo="$1"
output="$2"
cd "$repo"
count="$(find "$output/tiny_overfit" -name metrics.json -type f | wc -l)"
if [[ "$count" -ne 6 ]]; then
  echo "Attribution run is incomplete: found $count/6 metrics.json files." >&2
  exit 3
fi
mapfile -t logs < <(find slurm_logs -maxdepth 1 -type f -name 'stage_a_attr_*' -print)
echo "Validated 6/6 runs; downloading results and ${#logs[@]} log files." >&2
tar -czf - "$output" "${logs[@]}"
REMOTE_SCRIPT

tar -xzf "$ARCHIVE" -C "$DEST"
shasum -a 256 "$ARCHIVE" > "$ARCHIVE.sha256"

ATTRIBUTION_ROOT="$DEST/$REMOTE_OUTPUT"
python "$REPO_ROOT/keypoint_net/diagnostics/summarize_stage_a_attribution.py" \
  --baseline-root "$BASELINE_ROOT" \
  --attribution-root "$ATTRIBUTION_ROOT" \
  > "$DEST/summary_console.json"

SUMMARY="$ATTRIBUTION_ROOT/A0_ATTRIBUTION_SUMMARY.json"
CSV="$ATTRIBUTION_ROOT/A0_ATTRIBUTION_RUNS.csv"
if [[ ! -s "$SUMMARY" || ! -s "$CSV" ]]; then
  echo "Downloaded runs, but local summary artifacts were not created." >&2
  exit 4
fi

cat > "$DOWNLOAD_ROOT/STAGE_A_ATTRIBUTION_LATEST.txt" <<EOF
$DEST
EOF

echo "Saved and summarized on the Mac."
echo "Folder: $DEST"
echo "Summary: $SUMMARY"
echo "Runs table: $CSV"
