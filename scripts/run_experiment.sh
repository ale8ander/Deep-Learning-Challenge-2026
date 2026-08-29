#!/usr/bin/env bash
set -euo pipefail

# Edit this configuration block for each experiment.
EXPERIMENT_NAME="hybrid_3244_r8_qv_lr2e6_e1"
DATA_PATH="data/processed/hybrid_3244.jsonl"
LEARNING_RATE="2e-6"
LORA_RANK="8"
TARGET_MODULES="qv"
EPOCHS="1"
SEED="2026"

MAX_SEQ_LENGTH="2048"
MAX_MINUTES="40"
TRAIN_BATCH_SIZE="4"
GRADIENT_ACCUMULATION="2"
EVAL_BATCH_SIZE="16"
MAX_NEW_TOKENS="1024"
RETRY_MAX_NEW_TOKENS="2048"
RETRY_BATCH_SIZE="4"

RUN_TRAIN="1"
RUN_EVAL="1"
RUN_LEADERBOARD="0"  # Change to 1 only when a leaderboard submission is needed.

OUTPUT_DIR="checkpoints/${EXPERIMENT_NAME}"
ADAPTER_PATH="${OUTPUT_DIR}/final_adapter"
EVAL_OUTPUT="outputs/${EXPERIMENT_NAME}_eval200.jsonl"
LEADERBOARD_OUTPUT="outputs/${EXPERIMENT_NAME}_leaderboard.jsonl"
SUBMISSION_OUTPUT="submissions/submission_${EXPERIMENT_NAME}.csv"

if [[ "${RUN_TRAIN}" == "1" ]]; then
  PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  python scripts/train_qlora.py \
    --data "${DATA_PATH}" \
    --output-dir "${OUTPUT_DIR}" \
    --max-seq-length "${MAX_SEQ_LENGTH}" \
    --max-minutes "${MAX_MINUTES}" \
    --learning-rate "${LEARNING_RATE}" \
    --batch-size "${TRAIN_BATCH_SIZE}" \
    --gradient-accumulation "${GRADIENT_ACCUMULATION}" \
    --lora-rank "${LORA_RANK}" \
    --target-modules "${TARGET_MODULES}" \
    --epochs "${EPOCHS}" \
    --seed "${SEED}"
fi

if [[ "${RUN_EVAL}" == "1" ]]; then
  python scripts/baseline.py \
    --adapter-path "${ADAPTER_PATH}" \
    --output "${EVAL_OUTPUT}" \
    --limit 200 \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --retry-max-new-tokens "${RETRY_MAX_NEW_TOKENS}" \
    --retry-batch-size "${RETRY_BATCH_SIZE}" \
    --batch-size "${EVAL_BATCH_SIZE}"
fi

if [[ "${RUN_LEADERBOARD}" == "1" ]]; then
  python scripts/submit_baseline.py \
    --adapter-path "${ADAPTER_PATH}" \
    --output "${LEADERBOARD_OUTPUT}" \
    --submission "${SUBMISSION_OUTPUT}" \
    --batch-size "${EVAL_BATCH_SIZE}" \
    --max-new-tokens "${MAX_NEW_TOKENS}" \
    --retry-max-new-tokens "${RETRY_MAX_NEW_TOKENS}" \
    --retry-batch-size "${RETRY_BATCH_SIZE}"
fi
