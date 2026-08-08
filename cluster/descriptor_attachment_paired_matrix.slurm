#!/bin/bash
#SBATCH -J descriptor_pairs
#SBATCH -t 04:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH --array=0-5%6
#SBATCH -o /work/scratch/%u/fresh_roll_logs/%x_%A_%a.out
#SBATCH -e /work/scratch/%u/fresh_roll_logs/%x_%A_%a.err

set -euo pipefail

EXPECTED_COMMIT="${DESCRIPTOR_EXPECTED_COMMIT:?set DESCRIPTOR_EXPECTED_COMMIT}"
PROJECT_ROOT="${DESCRIPTOR_PROJECT_ROOT:?set DESCRIPTOR_PROJECT_ROOT}"
VENV_DIR="${DESCRIPTOR_VENV_DIR:?set DESCRIPTOR_VENV_DIR}"
MATRIX_ROOT="${DESCRIPTOR_MATRIX_ROOT:?set DESCRIPTOR_MATRIX_ROOT}"
EXPECTED_SCRIPT_SHA256="${DESCRIPTOR_SLURM_SCRIPT_SHA256:?set DESCRIPTOR_SLURM_SCRIPT_SHA256}"
EXPECTED_ENV_LOCK_SHA256="${DESCRIPTOR_ENV_LOCK_SHA256:?set DESCRIPTOR_ENV_LOCK_SHA256}"
ENV_LOCK="$VENV_DIR/FRESH_ROLL_ENVIRONMENT.json"

if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ \
   || ! "$EXPECTED_SCRIPT_SHA256" =~ ^[0-9a-f]{64}$ \
   || ! "$EXPECTED_ENV_LOCK_SHA256" =~ ^[0-9a-f]{64}$ ]]; then
  echo "invalid paired-matrix identity/hash input" >&2
  exit 2
fi
for REQUIRED_DIRECTORY in "$PROJECT_ROOT" "$VENV_DIR" "$MATRIX_ROOT"; do
  if [[ ! -d "$REQUIRED_DIRECTORY" ]]; then
    echo "required directory is missing: $REQUIRED_DIRECTORY" >&2
    exit 2
  fi
done
if [[ ! -f "$ENV_LOCK" || -L "$ENV_LOCK" ]]; then
  echo "environment lock is missing or symlinked" >&2
  exit 2
fi

ACTUAL_SCRIPT_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | cut -d ' ' -f 1)"
ACTUAL_ENV_LOCK_SHA256="$(sha256sum "$ENV_LOCK" | cut -d ' ' -f 1)"
if [[ "$ACTUAL_SCRIPT_SHA256" != "$EXPECTED_SCRIPT_SHA256" \
   || "$ACTUAL_ENV_LOCK_SHA256" != "$EXPECTED_ENV_LOCK_SHA256" ]]; then
  echo "paired-matrix script/environment hash differs" >&2
  exit 2
fi

module purge
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

DESCRIPTOR_ENV_LOCK="$ENV_LOCK" python - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path
import numpy
import torch

lock = json.loads(Path(os.environ["DESCRIPTOR_ENV_LOCK"]).read_text())
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
    [os.sys.executable, "-m", "pip", "freeze", "--all"], check=True,
    capture_output=True, text=True,
).stdout.encode("utf-8")
if hashlib.sha256(freeze).hexdigest() != lock["pip_freeze"]["sha256"]:
    raise RuntimeError("live pip freeze differs from environment lock")
PY

ACTUAL_COMMIT="$(git rev-parse HEAD)"
ACTUAL_BRANCH="$(git branch --show-current)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" \
   || "$ACTUAL_BRANCH" != "agent/representation-oracles-20260726" ]]; then
  echo "paired-matrix source identity differs" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "paired-matrix tracked source is modified" >&2
  exit 2
fi

echo "gate=descriptor_attachment_paired_matrix job_id=$SLURM_JOB_ID array_job_id=$SLURM_ARRAY_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname)"
echo "generic_gpu_request=1 node_count=$SLURM_JOB_NUM_NODES pair_runs_sequentially=2"
nvidia-smi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
srun python -B -u -m keypoint_net.run_descriptor_attachment_matrix task \
  --task-index "$SLURM_ARRAY_TASK_ID" \
  --matrix-root "$MATRIX_ROOT" \
  --expected-commit "$EXPECTED_COMMIT" \
  --slurm-script-sha256 "$ACTUAL_SCRIPT_SHA256" \
  --environment-lock-sha256 "$ACTUAL_ENV_LOCK_SHA256"
