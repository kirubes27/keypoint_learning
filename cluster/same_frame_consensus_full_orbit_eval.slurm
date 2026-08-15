#!/bin/bash
#SBATCH -J sfcons_eval
#SBATCH -t 02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem-per-cpu=4000
#SBATCH --constraint="avx512"
#SBATCH --array=0-1%2
#SBATCH -A l0003424
#SBATCH -o /work/scratch/%u/fresh_roll_logs/%x_%A_%a.out
#SBATCH -e /work/scratch/%u/fresh_roll_logs/%x_%A_%a.err

set -euo pipefail

EXPECTED_COMMIT="${SAME_FRAME_CONSENSUS_EXPECTED_COMMIT:?set SAME_FRAME_CONSENSUS_EXPECTED_COMMIT}"
PROJECT_ROOT="${SAME_FRAME_CONSENSUS_PROJECT_ROOT:?set SAME_FRAME_CONSENSUS_PROJECT_ROOT}"
EVAL_PYTHON="${SAME_FRAME_CONSENSUS_EVAL_PYTHON:?set SAME_FRAME_CONSENSUS_EVAL_PYTHON}"
PLOT_PYTHON="${SAME_FRAME_CONSENSUS_PLOT_PYTHON:?set SAME_FRAME_CONSENSUS_PLOT_PYTHON}"
DATA_ROOT="${SAME_FRAME_CONSENSUS_DATA_ROOT:?set SAME_FRAME_CONSENSUS_DATA_ROOT}"
OUTCOME_ROOT="${SAME_FRAME_CONSENSUS_OUTCOME_ROOT:?set SAME_FRAME_CONSENSUS_OUTCOME_ROOT}"
CHECKPOINT_ROOT="${SAME_FRAME_CONSENSUS_CHECKPOINT_ROOT:?set SAME_FRAME_CONSENSUS_CHECKPOINT_ROOT}"
OUTPUT_ROOT="${SAME_FRAME_CONSENSUS_OUTPUT_ROOT:?set SAME_FRAME_CONSENSUS_OUTPUT_ROOT}"
CALIBRATION="${SAME_FRAME_CONSENSUS_CALIBRATION:?set SAME_FRAME_CONSENSUS_CALIBRATION}"
CALIBRATION_SHA256="${SAME_FRAME_CONSENSUS_CALIBRATION_SHA256:?set SAME_FRAME_CONSENSUS_CALIBRATION_SHA256}"
EXPECTED_SCRIPT_SHA256="${SAME_FRAME_CONSENSUS_SLURM_SCRIPT_SHA256:?set SAME_FRAME_CONSENSUS_SLURM_SCRIPT_SHA256}"
SCRIPT_PATH="$(realpath "$PROJECT_ROOT/cluster/same_frame_consensus_full_orbit_eval.slurm")"
PAIR_INDEX="$PROJECT_ROOT/docs/decisions/2026-07-26/representation_oracle_splits/pairs/roll__world_z__forward__train.json"

[[ "$EXPECTED_COMMIT" =~ ^[0-9a-f]{40}$ ]] || exit 2
for VALUE in "$CALIBRATION_SHA256" "$EXPECTED_SCRIPT_SHA256"; do
  [[ "$VALUE" =~ ^[0-9a-f]{64}$ ]] || exit 2
done
for REQUIRED in "$PROJECT_ROOT" "$DATA_ROOT" "$OUTCOME_ROOT" "$CHECKPOINT_ROOT"; do
  [[ -d "$REQUIRED" ]] || { echo "missing directory: $REQUIRED" >&2; exit 2; }
done
for REQUIRED in "$CALIBRATION" "$SCRIPT_PATH" "$PAIR_INDEX"; do
  [[ -f "$REQUIRED" && ! -L "$REQUIRED" ]] || {
    echo "missing regular file: $REQUIRED" >&2
    exit 2
  }
done
for PYTHON_EXECUTABLE in "$EVAL_PYTHON" "$PLOT_PYTHON"; do
  [[ -x "$PYTHON_EXECUTABLE" ]] || exit 2
  [[ -f "$(readlink -f "$PYTHON_EXECUTABLE")" ]] || exit 2
done
[[ "$(sha256sum "$CALIBRATION" | cut -d ' ' -f 1)" == "$CALIBRATION_SHA256" ]] || exit 2
[[ "$(sha256sum "$SCRIPT_PATH" | cut -d ' ' -f 1)" == "$EXPECTED_SCRIPT_SHA256" ]] || exit 2

module purge
cd "$PROJECT_ROOT"
[[ "$(git rev-parse HEAD)" == "$EXPECTED_COMMIT" ]] || exit 2
[[ "$(git branch --show-current)" == "agent/sliding-wobble-eradication-20260814" ]] || exit 2
[[ -z "$(git status --porcelain --untracked-files=all)" ]] || {
  echo "remote checkout is not clean" >&2
  exit 2
}
"$EVAL_PYTHON" -c "import numpy, PIL, torch; assert torch.__version__ == '2.5.1+cu121'"
"$PLOT_PYTHON" -c "import matplotlib, numpy, PIL; assert matplotlib.__version__ == '3.9.2'"
"$EVAL_PYTHON" -c "from keypoint_net.same_frame_consensus import LOCKED_CONSENSUS_SPECS; assert len(LOCKED_CONSENSUS_SPECS) == 5"

case "${SLURM_ARRAY_TASK_ID:?missing Slurm array task ID}" in
  0) ARM=control ;;
  1) ARM=ocr_zncc ;;
  *) echo "Slurm array task is outside the two mature checkpoint arms" >&2; exit 2 ;;
esac

CELL="task80_assisted__${ARM}__seed42"
OUTCOME="$OUTCOME_ROOT/${CELL}.json"
CHECKPOINT="$CHECKPOINT_ROOT/${CELL}/best_model.pt"
NATIVE_DIR="$OUTPUT_ROOT/${CELL}__best_model__native"
CONSENSUS_DIR="$OUTPUT_ROOT/${CELL}__best_model__same_frame_consensus_v1"
VISUAL_DIR="$OUTPUT_ROOT/${CELL}__best_model__native_vs_consensus_visuals"

for REQUIRED in "$OUTCOME" "$CHECKPOINT"; do
  [[ -f "$REQUIRED" && ! -L "$REQUIRED" ]] || {
    echo "missing regular input file: $REQUIRED" >&2
    exit 2
  }
done
for DESTINATION in "$NATIVE_DIR" "$CONSENSUS_DIR" "$VISUAL_DIR"; do
  [[ ! -e "$DESTINATION" ]] || {
    echo "refusing to reuse output path: $DESTINATION" >&2
    exit 2
  }
done

mkdir -p "$OUTPUT_ROOT"
echo "gate=same_frame_consensus_full_orbit_frozen_eval arm=$ARM job_id=$SLURM_JOB_ID host=$(hostname)"

srun "$EVAL_PYTHON" -B -u -m keypoint_net.run_frozen_wobble_forensics \
  --outcome-json "$OUTCOME" \
  --checkpoint-role best_model \
  --checkpoint "$CHECKPOINT" \
  --decoder native \
  --data-root "$DATA_ROOT" \
  --pair-index "$PAIR_INDEX" \
  --calibration "$CALIBRATION" \
  --expected-calibration-sha256 "$CALIBRATION_SHA256" \
  --output-dir "$NATIVE_DIR"

srun "$EVAL_PYTHON" -B -u -m keypoint_net.run_frozen_wobble_forensics \
  --outcome-json "$OUTCOME" \
  --checkpoint-role best_model \
  --checkpoint "$CHECKPOINT" \
  --decoder same_frame_consensus_v1 \
  --data-root "$DATA_ROOT" \
  --pair-index "$PAIR_INDEX" \
  --calibration "$CALIBRATION" \
  --expected-calibration-sha256 "$CALIBRATION_SHA256" \
  --output-dir "$CONSENSUS_DIR"

srun "$PLOT_PYTHON" -B -u -m keypoint_net.plot_frozen_wobble_pair \
  --first-name native \
  --first-run "$NATIVE_DIR" \
  --second-name consensus \
  --second-run "$CONSENSUS_DIR" \
  --data-root "$DATA_ROOT" \
  --output-dir "$VISUAL_DIR"
