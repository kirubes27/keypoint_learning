#!/bin/bash
#SBATCH -J hammer_sweep
#SBATCH -t 12:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --array=1-432%2
#SBATCH -o /work/scratch/%u/phd_phase_a/slurm_logs/%x_%A_%a.out
#SBATCH -e /work/scratch/%u/phd_phase_a/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="${PROJECT_ROOT:-/work/scratch/$USER/phd_phase_a}"
VENV_DIR="${VENV_DIR:-/work/scratch/$USER/venvs/phd_keypoint}"
DATA_ROOT="${DATA_ROOT:-$PROJECT_ROOT/tdw_phase_a_starter /_tdw_world_z_roll_base_panel_512_v2}"
RUNS_DIR="${RUNS_DIR:-$PROJECT_ROOT/keypoint_net/runs_full_sweep_hammer_shared_affine}"
FILTER_FILE="${FILTER_FILE:-$PROJECT_ROOT/cluster/hammer_sweep_filters.txt}"

mkdir -p "$PROJECT_ROOT/slurm_logs" "$RUNS_DIR"

module purge
module load gcc openmpi
module load python
source "$VENV_DIR/bin/activate"

cd "$PROJECT_ROOT"

if [[ ! -f "$FILTER_FILE" ]]; then
  python cluster/make_hammer_sweep_filters.py "$FILTER_FILE"
fi

FILTER="$(sed -n "${SLURM_ARRAY_TASK_ID}p" "$FILTER_FILE")"
if [[ -z "$FILTER" ]]; then
  echo "No filter for SLURM_ARRAY_TASK_ID=$SLURM_ARRAY_TASK_ID" >&2
  exit 2
fi

echo "array_job_id=$SLURM_ARRAY_JOB_ID"
echo "array_task_id=$SLURM_ARRAY_TASK_ID"
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
