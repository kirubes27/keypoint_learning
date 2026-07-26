#!/bin/bash
#SBATCH -J d23_full
#SBATCH -t 01:00:00
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=5000
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --array=0-8%2
#SBATCH -o /work/scratch/%u/keypoint_preoperator_logs/%x_%A_%a.out
#SBATCH -e /work/scratch/%u/keypoint_preoperator_logs/%x_%A_%a.err

set -euo pipefail

EXPECTED_COMMIT="${DECISION23_EXPECTED_COMMIT:?set DECISION23_EXPECTED_COMMIT}"
D1_REPORT="${DECISION23_D1_REPORT:?set DECISION23_D1_REPORT}"
BASE_PROJECT_ROOT="/work/scratch/$USER/keypoint_preoperator_gates_20260726"
TASK_CLONE_ROOT="/work/scratch/$USER/keypoint_preoperator_task_clones"
PROJECT_ROOT="$TASK_CLONE_ROOT/d2_${EXPECTED_COMMIT}_${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
DATA_ROOT="/work/scratch/$USER/keypoint_learning/_tdw_world_z_roll_base_panel_512_v2"
VENV_DIR="/work/scratch/$USER/venvs/keypoint_learning"
OUTPUT_ROOT="/work/scratch/$USER/keypoint_preoperator_runs/decision23_$EXPECTED_COMMIT"
SCRIPT="keypoint_net/diagnostics/decision23_diagnostic_head.py"

ARMS=(raw_linear raw_linear raw_linear probability_linear probability_linear probability_linear fixed_expectation fixed_expectation fixed_expectation)
SEEDS=(42 43 44 42 43 44 42 43 44)
ARM="${ARMS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

module purge
module --ignore_cache load gcc
module --ignore_cache load openmpi
module load python
source "$VENV_DIR/bin/activate"
mkdir -p "$TASK_CLONE_ROOT"
if [[ -e "$PROJECT_ROOT" && ! -d "$PROJECT_ROOT/.git" ]]; then
  echo "task clone path exists but is not a Git clone: $PROJECT_ROOT" >&2
  exit 2
fi
if [[ ! -d "$PROJECT_ROOT/.git" ]]; then
  git clone --no-hardlinks --branch agent/preoperator-gates-20260726 \
    --single-branch "$BASE_PROJECT_ROOT" "$PROJECT_ROOT"
fi
cd "$PROJECT_ROOT"

ACTUAL_COMMIT="$(git rev-parse HEAD)"
ACTUAL_BRANCH="$(git branch --show-current)"
if [[ "$ACTUAL_COMMIT" != "$EXPECTED_COMMIT" ]]; then
  echo "commit mismatch: $ACTUAL_COMMIT != $EXPECTED_COMMIT" >&2
  exit 2
fi
if [[ "$ACTUAL_BRANCH" != "agent/preoperator-gates-20260726" ]]; then
  echo "branch mismatch: $ACTUAL_BRANCH" >&2
  exit 2
fi
if [[ -n "$(git status --porcelain)" ]]; then
  echo "worktree is dirty" >&2
  git status --short >&2
  exit 2
fi
if [[ ! -f "$D1_REPORT" ]]; then
  echo "missing D1 report: $D1_REPORT" >&2
  exit 2
fi
echo "gate=D2 job_id=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname)"
echo "branch=$ACTUAL_BRANCH commit=$ACTUAL_COMMIT arm=$ARM seed=$SEED"
nvidia-smi

RUN_DIR="$OUTPUT_ROOT/full/runs/e2e_${ARM}_standard64_k10_seed${SEED}"
RESUME_ARGS=()
if [[ -f "$RUN_DIR/training_summary.json" ]]; then
  echo "run already complete: $RUN_DIR"
  exit 0
fi
if [[ -e "$RUN_DIR/config.json" ]]; then
  if [[ ! -f "$RUN_DIR/last_checkpoint.pt" ]]; then
    echo "incomplete run has no resume checkpoint: $RUN_DIR" >&2
    exit 2
  fi
  RESUME_ARGS=(--resume)
fi

srun python -u "$SCRIPT" train \
  --data-root "$DATA_ROOT" \
  --output-root "$OUTPUT_ROOT" \
  --device cuda \
  --arm "$ARM" \
  --seed "$SEED" \
  --run-scope full \
  --d1-report "$D1_REPORT" \
  --batch-size 16 \
  --lr 1e-4 \
  --weight-decay 1e-5 \
  --min-epochs 1000 \
  --max-epochs 3000 \
  --eval-every 25 \
  --plateau-patience 400 \
  --relative-improvement 0.01 \
  "${RESUME_ARGS[@]}"
