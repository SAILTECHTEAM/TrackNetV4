#!/bin/bash

# Training script for VballNetV1c (GRU model) with 15 grayscale frames

# Set data paths
DATADIR=datasets/mix_volleyball_preprocessed
VAL_DATADIR=datasets/mix_volleyball_test_preprocessed

# Run training
uv run  train_gru.py \
  --data "$DATADIR" \
  --val_data "$VAL_DATADIR" \
  --model_name VballNetV1a \
  --seq 15 \
  --grayscale \
  --optimizer AdamW \
  --lr 0.001 \
  --epochs 200 \
  --batch 40  \
  --scheduler ReduceLROnPlateau \
  --workers 8

echo "Training completed!"
