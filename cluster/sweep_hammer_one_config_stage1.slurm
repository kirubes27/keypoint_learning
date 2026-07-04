#!/bin/bash
#SBATCH -J hammer_gate
#SBATCH -t 12:00:00
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
RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/keypoint_net/runs_gate_hammer_shared_affine}"
FILTER="lambda_act=1.0,lambda_cycle=0.1,lambda_disp=0.1,lambda_ent=0.01,lambda_inv=0.1,lambda_loc=0.0,lambda_smooth=0.0"

mkdir -p "$PROJECT_ROOT/slurm_logs" "$RUNS_DIR"

module purge
module load gcc openmpi
module load python
source "$VENV_DIR/bin/activate"

cd "$PROJECT_ROOT"

echo "job_id=$SLURM_JOB_ID"
echo "host=$(hostname)"
echo "filter=$FILTER"
nvidia-smi

srun python -u keypoint_net/sweep.py \
  --data_root "$DATA_ROOT" \
  --object engineers_hammer_vray \
  --operator_type shared_affine \
  --filter "$FILTER" \
  --stage 1 \
  --output_dir "$RUNS_DIR"
