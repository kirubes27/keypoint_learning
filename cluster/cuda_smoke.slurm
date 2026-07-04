#!/bin/bash
#SBATCH -J cuda_smoke
#SBATCH -t 00:10:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=2000
#SBATCH -n 1
#SBATCH -c 4
#SBATCH -o /work/scratch/%u/phd_phase_a/slurm_logs/%x.out.%j
#SBATCH -e /work/scratch/%u/phd_phase_a/slurm_logs/%x.err.%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/scratch/$USER/phd_phase_a}"
VENV_DIR="${VENV_DIR:-/work/scratch/$USER/venvs/phd_keypoint}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/tdw_phase_a_starter /_tdw_world_z_roll_base_panel_512_v2}"

mkdir -p "$PROJECT_ROOT/slurm_logs"

module purge
module load gcc openmpi
module load python
source "$VENV_DIR/bin/activate"

cd "$PROJECT_ROOT"

echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "data_root=$DATA_ROOT"
nvidia-smi

srun python - <<'PY'
import os
import sys
from pathlib import Path

import torch

project_root = Path(os.environ.get("PROJECT_ROOT", f"/work/scratch/{os.environ['USER']}/phd_phase_a"))
data_root = Path(os.environ["DATA_ROOT"])
sys.path.insert(0, str(project_root / "keypoint_net"))

from dataset import IndexPairDataset

print("python", sys.version)
print("torch", torch.__version__)
print("cuda_available", torch.cuda.is_available())
assert torch.cuda.is_available(), "CUDA is not visible inside the Slurm GPU allocation"
print("gpu_name", torch.cuda.get_device_name(0))

index_path = data_root / "indices" / "pairs_skip3_cyclic.json"
assert data_root.exists(), f"Missing data_root: {data_root}"
assert (data_root / "dataset_index.json").exists(), "Missing dataset_index.json"
assert index_path.exists(), f"Missing pair index: {index_path}"

ds = IndexPairDataset(
    data_root=str(data_root),
    index_path=str(index_path),
    img_size=512,
    include_backward=True,
    object_name="engineers_hammer_vray",
)
sample = ds[0]
print("dataset_len", len(ds))
print("x_t_shape", tuple(sample["x_t"].shape))
print("x_t1_shape", tuple(sample["x_t1"].shape))
print("action_label", sample["action_label"])
PY
