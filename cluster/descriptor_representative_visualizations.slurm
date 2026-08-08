#!/bin/bash
#SBATCH -J descriptor_viz
#SBATCH -t 00:30:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=4000
#SBATCH -o /work/scratch/%u/fresh_roll_logs/%x_%j.out
#SBATCH -e /work/scratch/%u/fresh_roll_logs/%x_%j.err

set -euo pipefail

SOURCE_ROOT="${VIZ_SOURCE_ROOT:?set VIZ_SOURCE_ROOT}"
VENV_DIR="${VIZ_VENV_DIR:?set VIZ_VENV_DIR}"
MATRIX_ROOT="${VIZ_MATRIX_ROOT:?set VIZ_MATRIX_ROOT}"
OUTPUT_ROOT="${VIZ_OUTPUT_ROOT:?set VIZ_OUTPUT_ROOT}"
FRAMES_DIR="${VIZ_FRAMES_DIR:?set VIZ_FRAMES_DIR}"
SOURCE_COMMIT="${VIZ_SOURCE_COMMIT:?set VIZ_SOURCE_COMMIT}"

if [[ "$(git -C "$SOURCE_ROOT" rev-parse HEAD)" != "$SOURCE_COMMIT" \
   || -n "$(git -C "$SOURCE_ROOT" status --porcelain --untracked-files=no)" ]]; then
  echo "historical source checkout differs" >&2
  exit 2
fi
for directory in "$VENV_DIR" "$MATRIX_ROOT/runs" "$FRAMES_DIR" "$OUTPUT_ROOT"; do
  [[ -d "$directory" ]] || { echo "missing directory: $directory" >&2; exit 2; }
done

module purge
source "$VENV_DIR/bin/activate"
cd "$SOURCE_ROOT"
echo "gate=descriptor_representative_visualizations job=$SLURM_JOB_ID host=$(hostname)"
echo "training_performed=false generic_gpu_request=1 node_count=$SLURM_JOB_NUM_NODES"
nvidia-smi

for ARM in control attachment; do
  CELL_ID="task80_assisted__${ARM}__seed42"
  CHECKPOINT="$MATRIX_ROOT/runs/$CELL_ID/best_model.pt"
  CELL_OUTPUT="$OUTPUT_ROOT/$CELL_ID"
  [[ -f "$CHECKPOINT" ]] || { echo "missing checkpoint: $CHECKPOINT" >&2; exit 2; }
  [[ ! -e "$CELL_OUTPUT" ]] || { echo "visualization output exists: $CELL_OUTPUT" >&2; exit 2; }

  "$VENV_DIR/bin/python" -B -u keypoint_net/ablations.py \
    --checkpoint "$CHECKPOINT" --frames_dir "$FRAMES_DIR" \
    --img_size 512 --frame_skip 3 --seed 42 \
    --output_dir "$CELL_OUTPUT/ablations"
  "$VENV_DIR/bin/python" -B -u keypoint_net/eval_compositionality.py \
    --checkpoint "$CHECKPOINT" --frames_dir "$FRAMES_DIR" \
    --img_size 512 --max_k 10 \
    --output_dir "$CELL_OUTPUT/compositionality"
  "$VENV_DIR/bin/python" -B -u keypoint_net/visualize.py \
    --checkpoint "$CHECKPOINT" --frames_dir "$FRAMES_DIR" \
    --img_size 512 --output_dir "$CELL_OUTPUT/visualizations"
done
