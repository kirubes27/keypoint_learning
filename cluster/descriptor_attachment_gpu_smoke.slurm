#!/bin/bash
#SBATCH -J descriptor_gpu_smoke
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH -o /work/scratch/%u/fresh_roll_logs/%x_%j.out
#SBATCH -e /work/scratch/%u/fresh_roll_logs/%x_%j.err

set -euo pipefail

EXPECTED_COMMIT="${DESCRIPTOR_EXPECTED_COMMIT:?set DESCRIPTOR_EXPECTED_COMMIT}"
PROJECT_ROOT="${DESCRIPTOR_PROJECT_ROOT:?set DESCRIPTOR_PROJECT_ROOT}"
VENV_DIR="${DESCRIPTOR_VENV_DIR:?set DESCRIPTOR_VENV_DIR}"
MANIFEST="${DESCRIPTOR_EXPERIMENT_MANIFEST:?set DESCRIPTOR_EXPERIMENT_MANIFEST}"
MANIFEST_SHA256="${DESCRIPTOR_EXPERIMENT_MANIFEST_SHA256:?set DESCRIPTOR_EXPERIMENT_MANIFEST_SHA256}"
WEIGHT_RECEIPT="${DESCRIPTOR_WEIGHT_RECEIPT:?set DESCRIPTOR_WEIGHT_RECEIPT}"
WEIGHT_RECEIPT_SHA256="${DESCRIPTOR_WEIGHT_RECEIPT_SHA256:?set DESCRIPTOR_WEIGHT_RECEIPT_SHA256}"
OUTPUT="${DESCRIPTOR_GPU_SMOKE_OUTPUT:?set DESCRIPTOR_GPU_SMOKE_OUTPUT}"
EXPECTED_SCRIPT_SHA256="${DESCRIPTOR_SLURM_SCRIPT_SHA256:?set DESCRIPTOR_SLURM_SCRIPT_SHA256}"
EXPECTED_ENV_LOCK_SHA256="${DESCRIPTOR_ENV_LOCK_SHA256:?set DESCRIPTOR_ENV_LOCK_SHA256}"
ENV_LOCK="$VENV_DIR/FRESH_ROLL_ENVIRONMENT.json"

for VALUE in "$MANIFEST_SHA256" "$WEIGHT_RECEIPT_SHA256" \
             "$EXPECTED_SCRIPT_SHA256" "$EXPECTED_ENV_LOCK_SHA256"; do
  if [[ ! "$VALUE" =~ ^[0-9a-f]{64}$ ]]; then
    echo "invalid SHA-256 argument" >&2
    exit 2
  fi
done
if [[ ! "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]]; then
  echo "invalid expected commit" >&2
  exit 2
fi
for REQUIRED_DIRECTORY in "$PROJECT_ROOT" "$VENV_DIR"; do
  if [[ ! -d "$REQUIRED_DIRECTORY" ]]; then
    echo "required directory is missing: $REQUIRED_DIRECTORY" >&2
    exit 2
  fi
done
for REQUIRED_FILE in "$MANIFEST" "$WEIGHT_RECEIPT" "$ENV_LOCK"; do
  if [[ ! -f "$REQUIRED_FILE" || -L "$REQUIRED_FILE" ]]; then
    echo "required regular file is missing or symlinked: $REQUIRED_FILE" >&2
    exit 2
  fi
done
if [[ -e "$OUTPUT" ]]; then
  echo "GPU smoke output already exists: $OUTPUT" >&2
  exit 2
fi

ACTUAL_SCRIPT_SHA256="$(sha256sum "${BASH_SOURCE[0]}" | cut -d ' ' -f 1)"
ACTUAL_ENV_LOCK_SHA256="$(sha256sum "$ENV_LOCK" | cut -d ' ' -f 1)"
ACTUAL_MANIFEST_SHA256="$(sha256sum "$MANIFEST" | cut -d ' ' -f 1)"
ACTUAL_WEIGHT_RECEIPT_SHA256="$(sha256sum "$WEIGHT_RECEIPT" | cut -d ' ' -f 1)"
if [[ "$ACTUAL_SCRIPT_SHA256" != "$EXPECTED_SCRIPT_SHA256" \
   || "$ACTUAL_ENV_LOCK_SHA256" != "$EXPECTED_ENV_LOCK_SHA256" \
   || "$ACTUAL_MANIFEST_SHA256" != "$MANIFEST_SHA256" \
   || "$ACTUAL_WEIGHT_RECEIPT_SHA256" != "$WEIGHT_RECEIPT_SHA256" ]]; then
  echo "hash-bound GPU smoke input differs" >&2
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
    [os.sys.executable, "-m", "pip", "freeze", "--all"],
    check=True, capture_output=True, text=True,
).stdout.encode("utf-8")
if hashlib.sha256(freeze).hexdigest() != lock["pip_freeze"]["sha256"]:
    raise RuntimeError("live pip freeze differs from environment lock")
PY

ACTUAL_COMMIT="$(git rev-parse HEAD)"
ACTUAL_BRANCH="$(git branch --show-current)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" \
   || "$ACTUAL_BRANCH" != "agent/representation-oracles-20260726" ]]; then
  echo "GPU smoke source identity differs" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain --untracked-files=no)" ]]; then
  echo "GPU smoke tracked worktree is dirty" >&2
  exit 2
fi

echo "gate=descriptor_attachment_gpu_smoke job_id=$SLURM_JOB_ID host=$(hostname)"
echo "branch=$ACTUAL_BRANCH commit=$ACTUAL_COMMIT"
echo "generic_gpu_request=1 node_count=$SLURM_JOB_NUM_NODES"
nvidia-smi

export CUBLAS_WORKSPACE_CONFIG=:4096:8
srun python -B -u -m keypoint_net.diagnostics.run_descriptor_attachment_gpu_smoke \
  --expected-commit "$EXPECTED_COMMIT" \
  --experiment-manifest "$MANIFEST" \
  --experiment-manifest-sha256 "$MANIFEST_SHA256" \
  --weight-receipt "$WEIGHT_RECEIPT" \
  --weight-receipt-sha256 "$WEIGHT_RECEIPT_SHA256" \
  --output "$OUTPUT"
