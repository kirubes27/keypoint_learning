#!/bin/bash
#SBATCH -J stage_r2_pilot
#SBATCH -t 02:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH -n 1
#SBATCH -c 8
#SBATCH -o /work/scratch/%u/keypoint_learning/slurm_logs/%x_%j.out
#SBATCH -e /work/scratch/%u/keypoint_learning/slurm_logs/%x_%j.err

set -euo pipefail

PROJECT_ROOT="/work/scratch/$USER/keypoint_learning"
VENV_DIR="/work/scratch/$USER/venvs/keypoint_learning"
DATA_ROOT="$PROJECT_ROOT/_tdw_world_z_roll_base_panel_512_v2"
OUTPUT_ROOT="$PROJECT_ROOT/keypoint_net/diagnostics/outputs/final_material_keypoints/stage_r2_representative_pilot"
RUN_DIR="$OUTPUT_ROOT/runs/coordinate_standard64_k10_seed41"

mkdir -p "$PROJECT_ROOT/slurm_logs" "$OUTPUT_ROOT"
module purge
module --ignore_cache load gcc
module --ignore_cache load openmpi
module load python
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"

echo "experiment=stage_r2_representative_pilot job_id=$SLURM_JOB_ID"
echo "branch=$(git branch --show-current) commit=$(git rev-parse HEAD)"
nvidia-smi

srun python -u keypoint_net/diagnostics/stage_a_supervised_control.py train \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --num-keypoints 10 \
  --supervision coordinate \
  --shape-constraint none \
  --seed 41 \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --min-epochs 1000 \
  --max-epochs 3000 \
  --eval-every 25 \
  --plateau-patience 400 \
  --relative-improvement 0.01

python keypoint_net/diagnostics/summarize_representative_coordinate_pilot.py \
  --run-dir "$RUN_DIR" \
  --output "$OUTPUT_ROOT/REPRESENTATIVE_PILOT_SUMMARY.json"
