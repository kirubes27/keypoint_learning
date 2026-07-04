#!/bin/bash
#SBATCH -J day45_full
#SBATCH -t 06:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --array=0-3%2
#SBATCH -o /work/scratch/%u/keypoint_learning/slurm_logs/%x_%A_%a.out
#SBATCH -e /work/scratch/%u/keypoint_learning/slurm_logs/%x_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/work/scratch/$USER/keypoint_learning"
VENV_DIR="/work/scratch/$USER/venvs/keypoint_learning"
DATA_ROOT="$PROJECT_ROOT/_tdw_world_z_roll_base_panel_512_v2"
ARMS=(coordinate coordinate heatmap heatmap)
SEEDS=(42 43 42 43)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "$PROJECT_ROOT/slurm_logs" "$PROJECT_ROOT/keypoint_net/diagnostics/outputs/day45_runs"
module purge
module load gcc openmpi
module load python
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

echo "job_id=$SLURM_JOB_ID host=$(hostname) arm=$ARM seed=$SEED"
nvidia-smi
srun python -u keypoint_net/diagnostics/day45_supervised_control.py \
  --data-root "$DATA_ROOT" \
  --arm "$ARM" \
  --heatmap-res 64 \
  --epochs 500 \
  --eval-every 25 \
  --batch-size 16 \
  --seed "$SEED" \
  --output-root "$PROJECT_ROOT/keypoint_net/diagnostics/outputs/day45_runs"
