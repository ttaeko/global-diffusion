#!/usr/bin/env bash
set -euo pipefail

DATA="data/alps_curriculum_10m_hydro_v4.h5"
OUTPUT="runs/one_pass_10m_candidate_a"

EPOCHS=50
BATCH_SIZE=2
LR="2e-4"
CROP_SIZE=128
NUM_WORKERS=0

SAVE_EVERY=5
RENDER_EVERY=1
RENDER_COUNT=3

MAX_TRAIN_BATCHES=""
MAX_VAL_BATCHES=""
RESUME=""

GRADIENT_WEIGHT="0.0"

PYTHON="${PYTHON:-python}"

while [[ $# -gt 0 ]]; do
    case "$1" in
        --data)
            DATA="$2"
            shift 2
            ;;

        --output)
            OUTPUT="$2"
            shift 2
            ;;

        --epochs)
            EPOCHS="$2"
            shift 2
            ;;

        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;

        --lr)
            LR="$2"
            shift 2
            ;;

        --crop-size)
            CROP_SIZE="$2"
            shift 2
            ;;

        --num-workers)
            NUM_WORKERS="$2"
            shift 2
            ;;

        --save-every)
            SAVE_EVERY="$2"
            shift 2
            ;;

        --render-every)
            RENDER_EVERY="$2"
            shift 2
            ;;

        --render-count)
            RENDER_COUNT="$2"
            shift 2
            ;;

        --max-train-batches)
            MAX_TRAIN_BATCHES="$2"
            shift 2
            ;;

        --max-val-batches)
            MAX_VAL_BATCHES="$2"
            shift 2
            ;;

        --resume)
            RESUME="$2"
            shift 2
            ;;
        
        --gradient-weight)
            GRADIENT_WEIGHT="$2"
            shift 2
            ;;

        *)
            echo "Unknown argument: $1"
            exit 1
            ;;
    esac
done

CMD=(
    "$PYTHON"
    -m terrain_diffusion.training.train_one_pass_10m

    --data "$DATA"
    --output-dir "$OUTPUT"

    --epochs "$EPOCHS"
    --batch-size "$BATCH_SIZE"
    --lr "$LR"
    --crop-size-30m "$CROP_SIZE"
    --num-workers "$NUM_WORKERS"

    --save-every "$SAVE_EVERY"
    --render-every "$RENDER_EVERY"
    --render-count "$RENDER_COUNT"
    --gradient-weight "$GRADIENT_WEIGHT"
)

if [[ -n "$MAX_TRAIN_BATCHES" ]]; then
    CMD+=(
        --max-train-batches
        "$MAX_TRAIN_BATCHES"
    )
fi

if [[ -n "$MAX_VAL_BATCHES" ]]; then
    CMD+=(
        --max-val-batches
        "$MAX_VAL_BATCHES"
    )
fi

if [[ -n "$RESUME" ]]; then
    CMD+=(
        --resume
        "$RESUME"
    )
fi

echo "Running:"
printf ' %q' "${CMD[@]}"
echo

"${CMD[@]}"