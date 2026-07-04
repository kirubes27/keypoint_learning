#!/usr/bin/env bash
set -euo pipefail

# Safe additive sync for the existing Lichtenberg layout. This intentionally
# has no --delete and does not transfer the dataset or historical run folders.
REMOTE_HOST="${REMOTE_HOST:-lcluster13.hrz.tu-darmstadt.de}"
REMOTE_USER="${REMOTE_USER:-ko75kamy}"
SSH_KEY="${SSH_KEY:-$HOME/.ssh/id_ed25519_lichtenberg}"
REMOTE_ROOT="/work/scratch/$REMOTE_USER/keypoint_learning"

ssh -i "$SSH_KEY" "$REMOTE_USER@$REMOTE_HOST" \
  "mkdir -p '$REMOTE_ROOT/cluster' '$REMOTE_ROOT/keypoint_net/diagnostics' '$REMOTE_ROOT/slurm_logs'"

rsync -az -e "ssh -i $SSH_KEY" \
  keypoint_net/model.py \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ROOT/keypoint_net/"

rsync -az -e "ssh -i $SSH_KEY" \
  keypoint_net/diagnostics/day45_supervised_control.py \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ROOT/keypoint_net/diagnostics/"

rsync -az -e "ssh -i $SSH_KEY" \
  cluster/day45_supervised_smoke.slurm \
  cluster/day45_supervised_full.slurm \
  "$REMOTE_USER@$REMOTE_HOST:$REMOTE_ROOT/cluster/"

echo "Synced Day-45 code without deleting remote data or runs."
