#!/usr/bin/env bash
set -euo pipefail

MODEL_ROOT="/vast/users/xiaodan/haozezhao/model"
DATA_ROOT="/vast/users/xiaodan/haozezhao/data"
MODEL_PATH="${MODEL_PATH:-${MODEL_ROOT}/checkpoints/qwen25vl_sft}"
NPROC="${NPROC:-4}"
MASTER_PORT="${MASTER_PORT:-33099}"
MIN_PIXELS="${MIN_PIXELS:-3136}"
MAX_PIXELS="${MAX_PIXELS:-2000000}"
DATASETS=("HR_Bench_4K" "HR_Bench_8K")

mkdir -p "${MODEL_ROOT}/eval/logs"

for DATASET in "${DATASETS[@]}"; do
  EVAL_JSON="eval/dataset/${DATASET}.jsonl"
  DATASET_PATH="${DATA_ROOT}/${DATASET}"
  OUTPUT_DIR="${MODEL_ROOT}/eval/${DATASET}"
  OUTPUT_FILE="${OUTPUT_DIR}/LaVCOT_predictions.jsonl"

  if [[ ! -f "${EVAL_JSON}" ]]; then
    echo "Missing evaluation file: ${EVAL_JSON}" >&2
    exit 1
  fi

  if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "Missing image directory: ${DATASET_PATH}" >&2
    exit 1
  fi

  mkdir -p "${OUTPUT_DIR}"
  python -m torch.distributed.run \
    --nproc_per_node="${NPROC}" \
    --master_port="${MASTER_PORT}" \
    eval/scripts/eval_LaVCOT.py \
    --model_path "${MODEL_PATH}" \
    --eval_dataset "${EVAL_JSON}" \
    --dataset_path "${DATASET_PATH}" \
    --output_file "${OUTPUT_FILE}" \
    --min_pixels "${MIN_PIXELS}" \
    --max_pixels "${MAX_PIXELS}" \
    2>&1 | tee "${MODEL_ROOT}/eval/logs/${DATASET}.log"
done
