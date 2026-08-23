#!/bin/bash

# Create one dedicated, pinned CUDA environment for the TAPNext++ 256 gate.
# The target must not exist. A failed attempt is preserved for audit and a new
# path must be chosen; this script never removes or overwrites an environment.

set -euo pipefail

VENV_DIR="${1:?usage: setup_tapnextpp_256_env.sh /new/absolute/venv/path}"
SETUP_SCRIPT="$(realpath "${BASH_SOURCE[0]}")"
if [[ "$VENV_DIR" != /* ]]; then
  echo "environment path must be absolute: $VENV_DIR" >&2
  exit 2
fi
if [[ -e "$VENV_DIR" ]]; then
  echo "environment path already exists: $VENV_DIR" >&2
  exit 2
fi

module purge
module --ignore_cache load gcc
module --ignore_cache load openmpi
module load python

python -m venv "$VENV_DIR"
"$VENV_DIR/bin/python" -m pip install --no-cache-dir \
  numpy==2.2.6 Pillow==12.1.0 einops==0.8.1
"$VENV_DIR/bin/python" -m pip install --no-cache-dir \
  torch==2.5.1 torchvision==0.20.1 \
  --index-url https://download.pytorch.org/whl/cu121

TAPNEXTPP_VENV_DIR="$VENV_DIR" \
TAPNEXTPP_ENV_SETUP_SCRIPT="$SETUP_SCRIPT" \
"$VENV_DIR/bin/python" - <<'PY'
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
from pathlib import Path

import numpy
import torch

venv = Path(os.environ["TAPNEXTPP_VENV_DIR"]).resolve(strict=True)
setup_script = Path(os.environ["TAPNEXTPP_ENV_SETUP_SCRIPT"]).resolve(strict=True)
versions = {
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
expected = {
    "python_implementation": "CPython",
    "python_version": "3.11.14",
    "torch": "2.5.1+cu121",
    "torchvision": "0.20.1+cu121",
    "einops": "0.8.1",
    "numpy": "2.2.6",
    "Pillow": "12.1.0",
    "pytorch_cuda": "12.1",
    "cudnn": 90100,
}
if versions != expected:
    raise RuntimeError(f"installed environment differs: {versions!r}")

freeze = subprocess.run(
    [str(venv / "bin/python"), "-m", "pip", "freeze", "--all"],
    check=True,
    capture_output=True,
    text=True,
).stdout
freeze_path = venv / "TAPNEXTPP_256_PIP_FREEZE.txt"
descriptor = os.open(freeze_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
try:
    os.write(descriptor, freeze.encode("utf-8"))
finally:
    os.close(descriptor)

document = {
    "schema_version": "tapnextpp_256_cluster_environment.v1",
    "artifact_type": "pinned_cluster_cuda_inference_environment",
    "versions": versions,
    "setup_script": {
        "absolute_path": str(setup_script),
        "sha256": hashlib.sha256(setup_script.read_bytes()).hexdigest(),
    },
    "pip_freeze": {
        "absolute_path": str(freeze_path),
        "sha256": hashlib.sha256(freeze.encode("utf-8")).hexdigest(),
    },
    "cuda_visible_during_login_node_setup": bool(torch.cuda.is_available()),
    "scope": "unchanged_environment_for_one_frozen_tapnextpp_256_gate",
}
data = (json.dumps(document, indent=2, sort_keys=True) + "\n").encode("utf-8")
lock_path = venv / "TAPNEXTPP_256_ENVIRONMENT.json"
descriptor = os.open(lock_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o444)
try:
    os.write(descriptor, data)
finally:
    os.close(descriptor)
print(lock_path)
print(hashlib.sha256(data).hexdigest())
PY
