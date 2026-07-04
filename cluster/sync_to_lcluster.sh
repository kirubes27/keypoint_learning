#!/usr/bin/env bash
set -euo pipefail

# Run this script on your Mac from /Users/kirubeso.r/Documents/PhD.

REMOTE_HOST="${REMOTE_HOST:-lcluster13.hrz.tu-darmstadt.de}"
REMOTE_USER="${REMOTE_USER:-ko75kamy}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_lichtenberg}"
REMOTE_ROOT="${REMOTE_ROOT:-/work/scratch/$REMOTE_USER/phd_phase_a}"

if [[ ! -f "$SSH_KEY" ]]; then
  echo "SSH key not found: $SSH_KEY" >&2
  exit 1
fi

echo "Creating remote directories: ${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}"
ssh -i "$SSH_KEY" "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '$REMOTE_ROOT/slurm_logs'"

rsync_common=(
  -az
  --delete
  -e "ssh -i $SSH_KEY"
  --exclude "__pycache__/"
  --exclude ".DS_Store"
  --exclude "keypoint_net/runs/"
  --exclude "keypoint_net/runs_local_smoke/"
  --exclude "runs/"
  --exclude "tmp/"
  --exclude ".git/"
)

echo "Syncing source code and docs..."
rsync "${rsync_common[@]}" \
  AGENTS.md CLAUDE.md docs keypoint_net cluster \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/"

echo "Syncing 512 roll dataset..."
ssh -i "$SSH_KEY" "${REMOTE_USER}@${REMOTE_HOST}" "mkdir -p '$REMOTE_ROOT/tdw_phase_a_starter '"
rsync -az --delete -e "ssh -i $SSH_KEY" \
  "tdw_phase_a_starter /_tdw_world_z_roll_base_panel_512_v2" \
  "${REMOTE_USER}@${REMOTE_HOST}:${REMOTE_ROOT}/tdw_phase_a_starter /"

echo "Done. Cluster root: ${REMOTE_ROOT}"
