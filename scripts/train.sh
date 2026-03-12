#!/usr/bin/env bash
python -m terrainfreespacenet.train \
  --data_dir data \
  --save_path checkpoints/terrainfreespacenet_best.pt \
  --num_points 2048 \
  --batch_size 16 \
  --epochs 50 \
  --lr 0.001