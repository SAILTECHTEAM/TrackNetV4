#!/bin/bash

# Training script for VballNetV1c (GRU model) with 15 grayscale frames

# Set data paths
DATADIR=datasets/mix_volleyball_preprocessed
VAL_DATADIR=datasets/mix_volleyball_test_preprocessed

# Run training
uv run  train_gru.py \
  --data "$DATADIR" \
  --val_data "$VAL_DATADIR" \
  --model_name VballNetV1c \
  --seq 15 \
  --grayscale \
  --optimizer AdamW \
  --lr 0.0005 \
  --epochs 200 \
  --batch 24  \
  --scheduler ReduceLROnPlateau \
  --workers 12 \
  --name exp_VballNetV1c_seq15_grayscale

echo "Training completed!"