#!/bin/bash
#SBATCH -J stage_r1_shape
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --array=0-2%3
#SBATCH -o /work/scratch/%u/keypoint_learning/slurm_logs/%x_%A_%a.out
#SBATCH -e /work/scratch/%u/keypoint_learning/slurm_logs/%x_%A_%a.err

set -euo pipefail

PROJECT_ROOT="/work/scratch/$USER/keypoint_learning"
VENV_DIR="/work/scratch/$USER/venvs/keypoint_learning"
DATA_ROOT="$PROJECT_ROOT/_tdw_world_z_roll_base_panel_512_v2"
OUTPUT_ROOT="$PROJECT_ROOT/keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r1_shape_gate"
SEEDS=(42 43 44)
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

mkdir -p "$PROJECT_ROOT/slurm_logs" "$OUTPUT_ROOT"
module purge
module --ignore_cache load gcc
module --ignore_cache load openmpi
module load python
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

echo "experiment=stage_r1_shape_gate job_id=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID seed=$SEED"
echo "branch=$(git branch --show-current) commit=$(git rev-parse HEAD)"
nvidia-smi

srun python -u keypoint_net/diagnostics/stage_a_supervised_control.py tiny-overfit \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --num-keypoints 10 \
  --supervision coordinate \
  --shape-constraint prediction_centered_js \
  --shape-weight 1.2128231385721024 \
  --shape-sigma-cells 1.0 \
  --seed "$SEED" \
  --tiny-max-steps 5000 \
  --tiny-eval-every 100 \
  --allow-gate-failure

