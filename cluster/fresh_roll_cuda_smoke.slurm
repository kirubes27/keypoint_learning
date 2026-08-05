#!/bin/bash
#SBATCH -J fresh_roll_cuda_smoke
#SBATCH -p acc_short
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:h100:1
#SBATCH --mem-per-cpu=5000
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o /work/scratch/%u/fresh_roll_logs/%x_%j.out
#SBATCH -e /work/scratch/%u/fresh_roll_logs/%x_%j.err

set -euo pipefail

EXPECTED_COMMIT="${FRESH_ROLL_EXPECTED_COMMIT:?set FRESH_ROLL_EXPECTED_COMMIT}"
PROJECT_ROOT="${FRESH_ROLL_PROJECT_ROOT:?set FRESH_ROLL_PROJECT_ROOT}"
DATA_ROOT="${FRESH_ROLL_DATA_ROOT:?set FRESH_ROLL_DATA_ROOT}"
VENV_DIR="${FRESH_ROLL_VENV_DIR:?set FRESH_ROLL_VENV_DIR}"
OUTPUT_ROOT="${FRESH_ROLL_OUTPUT_ROOT:?set FRESH_ROLL_OUTPUT_ROOT}"
EXPECTED_SCRIPT_SHA256="${FRESH_ROLL_SLURM_SCRIPT_SHA256:?set FRESH_ROLL_SLURM_SCRIPT_SHA256}"
EXPECTED_ENV_LOCK_SHA256="${FRESH_ROLL_ENV_LOCK_SHA256:?set FRESH_ROLL_ENV_LOCK_SHA256}"
ENV_LOCK="$VENV_DIR/FRESH_ROLL_ENVIRONMENT.json"
DIAGNOSTIC="keypoint_net/diagnostics/run_fresh_roll_cuda_smoke.py"

if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid expected commit" >&2
  exit 2
fi
if [[ ! "$EXPECTED_SCRIPT_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid expected Slurm script hash" >&2
  exit 2
fi
if [[ ! "$EXPECTED_ENV_LOCK_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid expected environment-lock hash" >&2
  exit 2
fi
for REQUIRED_PATH in "$PROJECT_ROOT" "$DATA_ROOT" "$VENV_DIR"; do
  if [[ ! -d "$REQUIRED_PATH" ]]; then
    echo "required directory is missing: $REQUIRED_PATH" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT_ROOT" ]]; then
  echo "fresh smoke output already exists: $OUTPUT_ROOT" >&2
  exit 2
fi
if [[ ! -f "$ENV_LOCK" || -L "$ENV_LOCK" ]]; then
  echo "environment lock is missing or is a symlink: $ENV_LOCK" >&2
  exit 2
fi

ACTUAL_SCRIPT_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | cut -d ' ' -f 1)"
ACTUAL_ENV_LOCK_SHA256="$(sha256sum "$ENV_LOCK" | cut -d ' ' -f 1)"
if [[ "$ACTUAL_SCRIPT_SHA256" != "$EXPECTED_SCRIPT_SHA256" ]]; then
  echo "Slurm script hash mismatch" >&2
  exit 2
fi
if [[ "$ACTUAL_ENV_LOCK_SHA256" != "$EXPECTED_ENV_LOCK_SHA256" ]]; then
  echo "environment lock hash mismatch" >&2
  exit 2
fi

module purge
module --ignore_cache load gcc
module --ignore_cache load openmpi
module load python
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

FRESH_ROLL_ENV_LOCK="$ENV_LOCK" python - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy
import torch

lock = json.loads(Path(os.environ["FRESH_ROLL_ENV_LOCK"]).read_text())
live = {
    "python_implementation": platform.python_implementation(),
    "python_version": platform.python_version(),
    "torch": str(torch.__version__),
    "torchvision": importlib.metadata.version("torchvision"),
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
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "commit mismatch: $ACTUAL_COMMIT != $EXPECTED_COMMIT" >&2
  exit 2
fi
if [[ "$ACTUAL_BRANCH" != "agent/representation-oracles-20260726" ]]; then
  echo "branch mismatch: $ACTUAL_BRANCH" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "tracked worktree is dirty" >&2
  git status --short --untracked-files=no >&2
  exit 2
fi

echo "gate=fresh_roll_cuda_smoke job_id=$SLURM_JOB_ID host=$(hostname)"
echo "branch=$ACTUAL_BRANCH commit=$ACTUAL_COMMIT"
echo "data_root=$DATA_ROOT"
echo "output_root=$OUTPUT_ROOT"
echo "slurm_script_sha256=$ACTUAL_SCRIPT_SHA256"
echo "environment_lock_sha256=$ACTUAL_ENV_LOCK_SHA256"
nvidia-smi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
export FRESH_ROLL_SLURM_SCRIPT_SHA256="$ACTUAL_SCRIPT_SHA256"
export FRESH_ROLL_ENV_LOCK_SHA256="$ACTUAL_ENV_LOCK_SHA256"
srun python -B -u "$DIAGNOSTIC" \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --expected-commit "$EXPECTED_COMMIT"
