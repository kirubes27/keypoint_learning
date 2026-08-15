#!/bin/bash
#SBATCH -J selfeq_eval
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32000
#SBATCH --constraint="avx512"
#SBATCH --array=0-1%2
#SBATCH -A l0003424
#SBATCH -o /work/scratch/%u/fresh_roll_logs/%x_%A_%a.out
#SBATCH -e /work/scratch/%u/fresh_roll_logs/%x_%A_%a.err

set -euo pipefail

EXPECTED_COMMIT="${SELF_EQUIVARIANCE_EVAL_EXPECTED_COMMIT:?set SELF_EQUIVARIANCE_EVAL_EXPECTED_COMMIT}"
PROJECT_ROOT="${SELF_EQUIVARIANCE_EVAL_PROJECT_ROOT:?set SELF_EQUIVARIANCE_EVAL_PROJECT_ROOT}"
VENV_DIR="${SELF_EQUIVARIANCE_EVAL_VENV_DIR:?set SELF_EQUIVARIANCE_EVAL_VENV_DIR}"
DATA_ROOT="${SELF_EQUIVARIANCE_EVAL_DATA_ROOT:?set SELF_EQUIVARIANCE_EVAL_DATA_ROOT}"
SMOKE_ROOT="${SELF_EQUIVARIANCE_EVAL_SMOKE_ROOT:?set SELF_EQUIVARIANCE_EVAL_SMOKE_ROOT}"
OUTPUT_ROOT="${SELF_EQUIVARIANCE_EVAL_OUTPUT_ROOT:?set SELF_EQUIVARIANCE_EVAL_OUTPUT_ROOT}"
CALIBRATION="${SELF_EQUIVARIANCE_EVAL_CALIBRATION:?set SELF_EQUIVARIANCE_EVAL_CALIBRATION}"
CALIBRATION_SHA256="${SELF_EQUIVARIANCE_EVAL_CALIBRATION_SHA256:?set SELF_EQUIVARIANCE_EVAL_CALIBRATION_SHA256}"
EXPECTED_SCRIPT_SHA256="${SELF_EQUIVARIANCE_EVAL_SLURM_SCRIPT_SHA256:?set SELF_EQUIVARIANCE_EVAL_SLURM_SCRIPT_SHA256}"
SCRIPT_PATH="$(realpath "$PROJECT_ROOT/cluster/same_frame_equivariance_full_orbit_eval.slurm")"
PAIR_INDEX="$PROJECT_ROOT/docs/decisions/2026-07-26/representation_oracle_splits/pairs/roll__world_z__forward__train.json"

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
for VALUE in "$CALIBRATION_SHA256" "$EXPECTED_SCRIPT_SHA256"; do
  [[ "$VALUE" =~ ^[0-9a-f]{64}$ ]] || exit 2
done
for REQUIRED in "$PROJECT_ROOT" "$VENV_DIR" "$DATA_ROOT" "$SMOKE_ROOT"; do
  [[ -d "$REQUIRED" ]] || { echo "missing directory: $REQUIRED" >&2; exit 2; }
done
for REQUIRED in "$CALIBRATION" "$SCRIPT_PATH" "$PAIR_INDEX"; do
  [[ -f "$REQUIRED" && ! -L "$REQUIRED" ]] || {
    echo "missing regular file: $REQUIRED" >&2
    exit 2
  }
done
[[ "$(sha256sum "$CALIBRATION" | cut -d ' ' -f 1)" == "$CALIBRATION_SHA256" ]] || exit 2
[[ "$(sha256sum "$SCRIPT_PATH" | cut -d ' ' -f 1)" == "$EXPECTED_SCRIPT_SHA256" ]] || exit 2

module purge
source "$VENV_DIR/bin/activate"
cd "$PROJECT_ROOT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || exit 2
[[ "$(git branch --show-current)" == "agent/sliding-wobble-eradication-20260814" ]] || exit 2
git diff --quiet HEAD -- || { echo "tracked source differs from HEAD" >&2; exit 2; }
git diff --cached --quiet || { echo "staged source differs from HEAD" >&2; exit 2; }

case "${SLURM_ARRAY_TASK_ID:?missing Slurm array task ID}" in
  0) RECIPE=task55_clean ;;
  1) RECIPE=task80_assisted ;;
  *) echo "Slurm array task is outside the two evaluation recipes" >&2; exit 2 ;;
esac

mkdir -p "$OUTPUT_ROOT"
echo "gate=same_frame_equivariance_full_orbit_frozen_eval recipe=$RECIPE job_id=$SLURM_JOB_ID host=$(hostname)"

for ROLE in best_model final_model; do
  for ARM in control self_equivariance; do
    CELL="${RECIPE}__${ARM}__seed42__smoke"
    RUN_DIR="$SMOKE_ROOT/$CELL"
    RECEIPT="$RUN_DIR/SELF_EQUIVARIANCE_COMPLETED_RUN_RECEIPT.json"
    CHECKPOINT="$RUN_DIR/${ROLE}.pt"
    DESTINATION="$OUTPUT_ROOT/${CELL}__${ROLE}"
    srun python -B -u -m keypoint_net.run_frozen_wobble_forensics \
      --smoke-completed-run-receipt "$RECEIPT" \
      --checkpoint-role "$ROLE" \
      --checkpoint "$CHECKPOINT" \
      --data-root "$DATA_ROOT" \
      --pair-index "$PAIR_INDEX" \
      --calibration "$CALIBRATION" \
      --expected-calibration-sha256 "$CALIBRATION_SHA256" \
      --output-dir "$DESTINATION"
  done

  srun python -B -u -m keypoint_net.plot_frozen_wobble_pair \
    --first-name control \
    --first-run "$OUTPUT_ROOT/${RECIPE}__control__seed42__smoke__${ROLE}" \
    --second-name self_equivariance \
    --second-run "$OUTPUT_ROOT/${RECIPE}__self_equivariance__seed42__smoke__${ROLE}" \
    --data-root "$DATA_ROOT" \
    --output-dir "$OUTPUT_ROOT/${RECIPE}__paired_visuals__${ROLE}"
done
