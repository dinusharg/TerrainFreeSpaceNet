#!/usr/bin/env bash
python -m terrainfreespacenet.evaluate \
  --data_dir data \
  --checkpoint checkpoints/terrainfreespacenet_best.pt \
  --batch_size 16