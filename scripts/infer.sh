#!/usr/bin/env bash
python -m terrainfreespacenet.infer \
  --input_csv examples/sample_input.csv \
  --checkpoint checkpoints/terrainfreespacenet_best.pt