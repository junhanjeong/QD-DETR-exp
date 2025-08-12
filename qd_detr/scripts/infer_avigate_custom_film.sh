#!/usr/bin/env bash
set -euo pipefail

# Inference launcher with FiLM enabled and optional gate logging flags.

PYTHON=${PYTHON:-python}
ARGS=("$@")

PYTHONPATH=$PYTHONPATH:. ${PYTHON} qd_detr/inference.py \
  --device 0 \
  --use_avigate_custom \
  --gating_type global \
  --mha_gate_temp 1.5 \
  --mha_gate_bias_init 1.0 \
  --mha_gate_scale_init 1.0 \
  --ffn_gate_alpha_init 0.3 \
  --ffn_gate_film \
  --ffn_gate_beta_init 0.3 \
  "${ARGS[@]}"

