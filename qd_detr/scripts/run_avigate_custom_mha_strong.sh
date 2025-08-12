#!/usr/bin/env bash
set -euo pipefail

# Training launcher variant with stronger MHA gate flow (higher bias).
# Useful if MHA gates tend to collapse near zero.

PYTHON=${PYTHON:-python}
ARGS=("$@")

PYTHONPATH=$PYTHONPATH:. ${PYTHON:-python} qd_detr/train.py \
  --device 0 \
  --n_epoch 10 \
  --bsz 16 \
  --eval_bsz 64 \
  --hidden_dim 256 \
  --fusion_layers 4 \
  --fusion_n_heads 4 \
  --use_avigate_custom \
  --gating_type global \
  --mha_gate_temp 1.5 \
  --mha_gate_bias_init 1.8 \
  --mha_gate_scale_init 1.0 \
  --ffn_gate_alpha_init 0.3 \
  "${ARGS[@]}"

