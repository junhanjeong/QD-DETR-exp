#!/usr/bin/env bash
# Wrapper to force using a specific split name and jsonl path even when
# TestOptions loads and overrides eval_split_name/eval_path from saved opt.json.
# Usage:
#   bash qd_detr/scripts/inference_force_split.sh \
#     <ckpt_path> <eval_split_name> [extra args passed to parser]
#
# Example:
#   bash qd_detr/scripts/inference_force_split.sh \
#     results/xxx/model_best.ckpt train_first1000 --gate_log

set -euo pipefail

ckpt_path=${1:?"ckpt path required"}
orig_ckpt_path="${ckpt_path}"
split_name=${2:?"eval split name required"}
shift 2

eval_path=data/highlight_${split_name}_release.jsonl
# If provided ckpt is empty or missing, try to fall back to a valid one in the same dir.
if [ ! -s "${ckpt_path}" ]; then
  ckpt_dir=$(dirname "${ckpt_path}")
  if [ -s "${ckpt_dir}/model_latest.ckpt" ]; then
    echo "[force-split] provided ckpt is empty; falling back to model_latest.ckpt"
    ckpt_path="${ckpt_dir}/model_latest.ckpt"
  else
    # pick the largest epoch ckpt if available
    alt_ckpt=$(ls -1 ${ckpt_dir}/model_e*.ckpt 2>/dev/null | head -n1 || true)
    if [ -n "${alt_ckpt}" ] && [ -s "${alt_ckpt}" ]; then
      echo "[force-split] provided ckpt is empty; falling back to ${alt_ckpt}"
      ckpt_path="${alt_ckpt}"
    else
      echo "[force-split][error] checkpoint not found or empty: ${1}"
      exit 1
    fi
  fi
fi

# Report effective checkpoint selection
if [ "${ckpt_path}" != "${orig_ckpt_path}" ]; then
  echo "[force-split] effective ckpt: ${ckpt_path} (from ${orig_ckpt_path})"
else
  echo "[force-split] ckpt: ${ckpt_path}"
fi

# Explicitly warn if not using model_best.ckpt
if [ "$(basename -- "${ckpt_path}")" != "model_best.ckpt" ]; then
  echo "[force-split] note: using non-best checkpoint -> $(basename -- "${ckpt_path}")"
fi

echo "[force-split] split: ${split_name}"
echo "[force-split] path:  ${eval_path}"
# Call start_inference programmatically so we can override split/path
# after TestOptions().parse() loads saved options.
# Ensure PYTHONPATH includes current repo without assuming it is pre-set.
export PYTHONPATH=".:${PYTHONPATH:-}"
python - --resume "${ckpt_path}" "$@" <<PY
from qd_detr.inference import start_inference
start_inference(split="${split_name}", splitfile="${eval_path}")
PY
