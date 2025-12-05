#!/bin/bash

# Script to train all VballNetV3c variants in parallel
# Usage: ./train_all_v3c.sh --data path/to/data --batch 4 --epochs 30
source  .venv/bin/activate

# Parse arguments
DATA_PATH=""
BATCH_SIZE=1
EPOCHS=30
OTHER_ARGS=""

while [[ $# -gt 0 ]]; do
  case $1 in
    --data)
      DATA_PATH="$2"
      shift 2
      ;;
    --batch)
      BATCH_SIZE="$2"
      shift 2
      ;;
    --epochs)
      EPOCHS="$2"
      shift 2
      ;;
    *)
      OTHER_ARGS="$OTHER_ARGS $1"
      shift
      ;;
  esac
done

# Check if data path is provided
if [ -z "$DATA_PATH" ]; then
  echo "Error: --data path is required"
  echo "Usage: ./train_all_v3c.sh --data path/to/data [--batch 1] [--epochs 30] [other_args]"
  exit 1
fi

echo "Training all VballNetV3c variants in parallel..."
echo "Data path: $DATA_PATH"
echo "Batch size: $BATCH_SIZE"
echo "Epochs: $EPOCHS"
echo "Other args: $OTHER_ARGS"
echo ""

# Function to train a specific variant
train_variant() {
  local variant=$1
  echo "Starting training for VballNetV3c ($variant)..."
  python train_v3c.py \
    --data "$DATA_PATH" \
    --model_variant "$variant" \
    --batch "$BATCH_SIZE" \
    --epochs "$EPOCHS" \
    --name "v3c_${variant}" \
    --grayscale  --seq 9 \
    $OTHER_ARGS
  echo "Finished training for VballNetV3c ($variant)"
}

# Train all variants in pa:rallel (background processes)
train_variant "original" &
PID1=$!

train_variant "minimal" &
PID2=$!

train_variant "improved" &
PID3=$!

# Enhanced and optimized variants might be slower, so we can run them separately if needed
# train_variant "optimized" &
# PID4=$!

# train_variant "enhanced" &
# PID5=$!

# Wait for all processes to complete
echo "Waiting for all training processes to complete..."
wait $PID1 $PID2 $PID3

echo "All VballNetV3c variants training completed!"
echo "Results are saved in the outputs/ directory with respective experiment names."
