#!/usr/bin/env bash
set -euo pipefail

# Minimal inference launcher for AVIGATE custom fusion with gating flags and gate logging.
# Required args usually include: --dset_name, --eval_path, feature dirs/dims, and --resume to load a checkpoint.
# Example:
#   bash qd_detr/scripts/infer_avigate_custom.sh \
#     --dset_name tvsum \
#     --eval_path data/tvsum/val.jsonl \
#     --v_feat_dirs data/vfeat_dir \
#     --t_feat_dir data/tfeat_dir \
#     --a_feat_dir data/afeat_dir \
#     --v_feat_dim 1024 --t_feat_dim 768 --a_feat_dim 512 \
#     --results_root results --exp_id avigate_sigmoid_tanh \
#     --resume results/exp_xxx/model_best.ckpt \
#     --gate_log --gate_sample_count 50

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
  "${ARGS[@]}"

