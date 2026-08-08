#!/bin/bash
#SBATCH -J descriptor_align
#SBATCH -t 01:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH --array=0-5%6
#SBATCH -o /work/scratch/%u/fresh_roll_logs/%x_%A_%a.out
#SBATCH -e /work/scratch/%u/fresh_roll_logs/%x_%A_%a.err

set -euo pipefail

SOURCE_ROOT="${ALIGN_SOURCE_ROOT:?set ALIGN_SOURCE_ROOT}"
AUDIT_ROOT="${ALIGN_AUDIT_ROOT:?set ALIGN_AUDIT_ROOT}"
VENV_DIR="${ALIGN_VENV_DIR:?set ALIGN_VENV_DIR}"
MATRIX_ROOT="${ALIGN_MATRIX_ROOT:?set ALIGN_MATRIX_ROOT}"
OUTPUT_ROOT="${ALIGN_OUTPUT_ROOT:?set ALIGN_OUTPUT_ROOT}"
DATA_ROOT="${ALIGN_DATA_ROOT:?set ALIGN_DATA_ROOT}"
SOURCE_COMMIT="${ALIGN_SOURCE_COMMIT:?set ALIGN_SOURCE_COMMIT}"
AUDIT_COMMIT="${ALIGN_AUDIT_COMMIT:?set ALIGN_AUDIT_COMMIT}"
SCRIPT_SHA256="${ALIGN_SLURM_SHA256:?set ALIGN_SLURM_SHA256}"
MODULE_SHA256="${ALIGN_MODULE_SHA256:?set ALIGN_MODULE_SHA256}"

MODULE="$AUDIT_ROOT/keypoint_net/descriptor_material_alignment.py"
SCRIPT="$AUDIT_ROOT/cluster/descriptor_material_alignment.slurm"
if [[ "$(sha256sum "$SCRIPT" | cut -d ' ' -f 1)" != "$SCRIPT_SHA256" \
   || "$(sha256sum "$MODULE" | cut -d ' ' -f 1)" != "$MODULE_SHA256" ]]; then
  echo "audit script/module hash differs" >&2
  exit 2
fi
if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" \
   || "$(git -C "$AUDIT_ROOT" rev-parse HEAD)" != "$AUDIT_COMMIT" ]]; then
  echo "source or audit commit differs" >&2
  exit 2
fi
if [[ -n "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=no)" \
   || -n "$(git -C "$AUDIT_ROOT" status --porcelain --untracked-files=no)" ]]; then
  echo "source or audit checkout has tracked changes" >&2
  exit 2
fi
for directory in "$VENV_DIR" "$MATRIX_ROOT/runs" "$MATRIX_ROOT/evaluations" \
                 "$OUTPUT_ROOT/results" "$DATA_ROOT"; do
  [[ -d "$directory" ]] || { echo "missing directory: $directory" >&2; exit 2; }
done

case "$SLURM_ARRAY_TASK_ID" in
  0) RECIPE="task55_clean"; SEED="42" ;;
  1) RECIPE="task55_clean"; SEED="43" ;;
  2) RECIPE="task55_clean"; SEED="44" ;;
  3) RECIPE="task80_assisted"; SEED="42" ;;
  4) RECIPE="task80_assisted"; SEED="43" ;;
  5) RECIPE="task80_assisted"; SEED="44" ;;
  *) echo "array task outside 0..5" >&2; exit 2 ;;
esac

module purge
source "$VENV_DIR/bin/activate"
cd "$SOURCE_ROOT"
echo "gate=descriptor_material_alignment job=$SLURM_JOB_ID task=$SLURM_ARRAY_TASK_ID host=$(hostname)"
echo "training_performed=false generic_gpu_request=1 node_count=$SLURM_JOB_NUM_NODES"
nvidia-smi

for ARM in control attachment; do
  CELL_ID="${RECIPE}__${ARM}__seed${SEED}"
  PYTHONPATH="$SOURCE_ROOT" "$VENV_DIR/bin/python" -B -u "$MODULE" cell \
    --source-repo-root "$SOURCE_ROOT" \
    --audit-repo-root "$AUDIT_ROOT" \
    --audit-commit "$AUDIT_COMMIT" \
    --receipt "$MATRIX_ROOT/runs/$CELL_ID/DESCRIPTOR_COMPLETED_RUN_RECEIPT.json" \
    --cell-id "$CELL_ID" \
    --data-root "$DATA_ROOT" \
    --evaluator-result "$MATRIX_ROOT/evaluations/$CELL_ID.json" \
    --output "$OUTPUT_ROOT/results/$CELL_ID.json" \
    --batch-size 7
done
