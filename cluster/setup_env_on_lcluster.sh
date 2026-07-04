#!/usr/bin/env bash
set -euo pipefail

# Run this on Lichtenberg after syncing the project.

PROJECT_ROOT="${PROJECT_ROOT:-/work/scratch/$USER/phd_phase_a}"
VENV_DIR="${VENV_DIR:-/work/scratch/$USER/venvs/phd_keypoint}"

mkdir -p "$PROJECT_ROOT/slurm_logs" "$VENV_DIR"

module purge
module load gcc openmpi
module load python

python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

python -m pip install --upgrade pip setuptools wheel
python -m pip install -r "$PROJECT_ROOT/keypoint_net/requirements.txt"

# Optional but useful for exact distance-to-mask rollout metrics.
python -m pip install scipy

python - <<'PY'
import sys
import torch
import torchvision
import numpy
import matplotlib
from PIL import Image

print("python", sys.version)
print("torch", torch.__version__)
print("torchvision", torchvision.__version__)
print("cuda_available", torch.cuda.is_available())
print("numpy", numpy.__version__)
print("matplotlib", matplotlib.__version__)
print("pillow", Image.__version__)
PY

echo "Environment ready: $VENV_DIR"
