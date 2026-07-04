#!/bin/bash
#SBATCH -J gpu_train_smoke
#SBATCH -t 00:30:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o /work/scratch/%u/phd_phase_a/slurm_logs/%x.out.%j
#SBATCH -e /work/scratch/%u/phd_phase_a/slurm_logs/%x.err.%j

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/scratch/$USER/phd_phase_a}"
VENV_DIR="${VENV_DIR:-/work/scratch/$USER/venvs/phd_keypoint}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/tdw_phase_a_starter /_tdw_world_z_roll_base_panel_512_v2}"
OUT_DIR="${OUT_DIR:-$PROJECT_ROOT/keypoint_net/runs_gpu_smoke}"
PAIRS_INDEX="$DATA_ROOT/indices/pairs_skip3_cyclic.json"

mkdir -p "$PROJECT_ROOT/slurm_logs" "$OUT_DIR"

module purge
module load gcc openmpi
module load python
source "$VENV_DIR/bin/activate"

cd "$PROJECT_ROOT"

echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "data_root=$DATA_ROOT"
nvidia-smi

srun python -u keypoint_net/train.py \
  --data_root "$DATA_ROOT" \
  --object engineers_hammer_vray \
  --pairs_index "$PAIRS_INDEX" \
  --operator_type shared_affine \
  --img_size 512 \
  --num_keypoints 10 \
  --epochs 1 \
  --batch_size 8 \
  --lr 1e-4 \
  --seed 42 \
  --lambda_act 1.0 \
  --lambda_cycle 0.1 \
  --lambda_disp 0.1 \
  --lambda_ent 0.01 \
  --lambda_inv 0.1 \
  --lambda_loc 0.0 \
  --lambda_smooth 0.0 \
  --num_action_classes 2 \
  --output_dir "$OUT_DIR"
