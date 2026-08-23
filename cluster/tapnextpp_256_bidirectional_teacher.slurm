#!/bin/bash
#SBATCH -J tapnextpp256_gate
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --constraint="nvd&avx512"
#SBATCH --mem-per-cpu=5000
#SBATCH -o /work/scratch/%u/tapnextpp_256_logs/%x_%j.out
#SBATCH -e /work/scratch/%u/tapnextpp_256_logs/%x_%j.err

set -euo pipefail

EXPECTED_COMMIT="${TAPNEXTPP_EXPECTED_COMMIT:?set TAPNEXTPP_EXPECTED_COMMIT}"
PROJECT_ROOT="${TAPNEXTPP_PROJECT_ROOT:?set TAPNEXTPP_PROJECT_ROOT}"
TAPNET_ROOT="${TAPNEXTPP_TAPNET_ROOT:?set TAPNEXTPP_TAPNET_ROOT}"
VENV_DIR="${TAPNEXTPP_VENV_DIR:?set TAPNEXTPP_VENV_DIR}"
CHECKPOINT="${TAPNEXTPP_CHECKPOINT:?set TAPNEXTPP_CHECKPOINT}"
MANIFEST="${TAPNEXTPP_MANIFEST:?set TAPNEXTPP_MANIFEST}"
INITIALS="${TAPNEXTPP_INITIALS:?set TAPNEXTPP_INITIALS}"
SEMANTIC_LOCK="${TAPNEXTPP_SEMANTIC_LOCK:?set TAPNEXTPP_SEMANTIC_LOCK}"
OBJECT_ROOT="${TAPNEXTPP_OBJECT_ROOT:?set TAPNEXTPP_OBJECT_ROOT}"
OUTPUT_DIR="${TAPNEXTPP_OUTPUT_DIR:?set TAPNEXTPP_OUTPUT_DIR}"
EXPECTED_SCRIPT_SHA256="${TAPNEXTPP_SLURM_SCRIPT_SHA256:?set TAPNEXTPP_SLURM_SCRIPT_SHA256}"
EXPECTED_ENV_LOCK_SHA256="${TAPNEXTPP_ENV_LOCK_SHA256:?set TAPNEXTPP_ENV_LOCK_SHA256}"
ENV_LOCK="$VENV_DIR/TAPNEXTPP_256_ENVIRONMENT.json"

if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ \
   || ! "$EXPECTED_SCRIPT_SHA256" =~ ^[0-9a-f]{64}$ \
   || ! "$EXPECTED_ENV_LOCK_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid commit or SHA-256 input" >&2
  exit 2
fi
for REQUIRED_DIRECTORY in "$PROJECT_ROOT" "$TAPNET_ROOT" "$VENV_DIR" "$OBJECT_ROOT"; do
  if [[ ! -d "$REQUIRED_DIRECTORY" ]]; then
    echo "required directory is missing: $REQUIRED_DIRECTORY" >&2
    exit 2
  fi
done
for REQUIRED_FILE in "$CHECKPOINT" "$MANIFEST" "$INITIALS" "$SEMANTIC_LOCK" "$ENV_LOCK"; do
  if [[ ! -f "$REQUIRED_FILE" || -L "$REQUIRED_FILE" ]]; then
    echo "required regular file is missing or symlinked: $REQUIRED_FILE" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_DIR" ]]; then
  echo "TAPNext++ output already exists: $OUTPUT_DIR" >&2
  exit 2
fi

ACTUAL_SCRIPT_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | cut -d ' ' -f 1)"
ACTUAL_ENV_LOCK_SHA256="$(sha256sum "$ENV_LOCK" | cut -d ' ' -f 1)"
if [[ "$ACTUAL_SCRIPT_SHA256" != "$EXPECTED_SCRIPT_SHA256" \
   || "$ACTUAL_ENV_LOCK_SHA256" != "$EXPECTED_ENV_LOCK_SHA256" ]]; then
  echo "Slurm script or environment-lock hash differs" >&2
  exit 2
fi

module purge
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

TAPNEXTPP_ENV_LOCK="$ENV_LOCK" python - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy
import torch

lock = json.loads(Path(os.environ["TAPNEXTPP_ENV_LOCK"]).read_text())
live = {
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "torch": str(torch.__version__),
    "torchvision": importlib.metadata.version("torchvision"),
    "einops": importlib.metadata.version("einops"),
    "numpy": str(numpy.__version__),
    "Pillow": importlib.metadata.version("Pillow"),
    "pytorch_cuda": torch.version.cuda,
    "cudnn": torch.backends.cudnn.version(),
}
if live != lock.get("versions"):
    raise RuntimeError(f"live environment differs from lock: {live!r}")
freeze = subprocess.run(
    [os.sys.executable, "-m", "pip", "freeze", "--all"],
    check=True,
    capture_output=True,
    text=True,
).stdout.encode("utf-8")
if hashlib.sha256(freeze).hexdigest() != lock["pip_freeze"]["sha256"]:
    raise RuntimeError("live pip freeze differs from environment lock")
PY

ACTUAL_COMMIT="$(git rev-parse HEAD)"
ACTUAL_BRANCH="$(git branch --show-current)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" \
   || "$ACTUAL_BRANCH" != "agent/material-transport-free-logits-20260823" ]]; then
  echo "TAPNext++ source identity differs" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "TAPNext++ source worktree is dirty" >&2
  git status --short >&2
  exit 2
fi

echo "gate=tapnextpp_256_bidirectional_teacher job_id=$SLURM_JOB_ID host=$(hostname)"
echo "branch=$ACTUAL_BRANCH commit=$ACTUAL_COMMIT"
echo "tapnet_root=$TAPNET_ROOT checkpoint=$CHECKPOINT"
echo "output_dir=$OUTPUT_DIR"
echo "slurm_script_sha256=$ACTUAL_SCRIPT_SHA256"
echo "environment_lock_sha256=$ACTUAL_ENV_LOCK_SHA256"
nvidia-smi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export PYTHONPATH="$TAPNET_ROOT:$PROJECT_ROOT${PYTHONPATH:+:$PYTHONPATH}"
srun python -B -u keypoint_net/run_tapnextpp_bidirectional_teacher.py \
  --manifest "$MANIFEST" \
  --initials "$INITIALS" \
  --semantic-lock "$SEMANTIC_LOCK" \
  --object-root "$OBJECT_ROOT" \
  --repo-root "$PROJECT_ROOT" \
  --tapnet-root "$TAPNET_ROOT" \
  --checkpoint "$CHECKPOINT" \
  --device cuda \
  --output-dir "$OUTPUT_DIR"
