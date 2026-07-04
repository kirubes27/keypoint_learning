#!/bin/bash
#SBATCH -J stage_a_k
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --array=0-11%4
#SBATCH -o /work/scratch/%u/keypoint_learning/slurm_logs/%x_%A_%a.out
#SBATCH -e /work/scratch/%u/keypoint_learning/slurm_logs/%x_%A_%a.err

set -euo pipefail
PROJECT_ROOT="/work/scratch/$USER/keypoint_learning"
VENV_DIR="/work/scratch/$USER/venvs/keypoint_learning"
DATA_ROOT="$PROJECT_ROOT/_tdw_world_z_roll_base_panel_512_v2"
OUTPUT_ROOT="$PROJECT_ROOT/keypoint_net/diagnostics/outputs/final_material_keypoints/stage_a_k_sweep"

KS=(5 5 5 10 10 10 15 15 15 20 20 20)
SEEDS=(42 43 44 42 43 44 42 43 44 42 43 44)
K="${KS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "$PROJECT_ROOT/slurm_logs" "$OUTPUT_ROOT"
module purge
module load gcc openmpi
module load python
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

echo "job_id=$SLURM_JOB_ID host=$(hostname) K=$K seed=$SEED"
nvidia-smi
srun python -u keypoint_net/diagnostics/stage_a_supervised_control.py tiny-overfit \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --num-keypoints "$K" \
  --seed "$SEED" \
  --tiny-max-steps 5000 \
  --tiny-eval-every 100 \
  --allow-gate-failure
