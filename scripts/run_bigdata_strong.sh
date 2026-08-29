#!/usr/bin/env bash
# 비어 있던 칸: "큰 데이터 + 강한 학습", 그리고 처음으로 추론과 동일한 system 프롬프트.
#
# 데이터: data/processed/unified_prompt_13k.jsonl (13,550건, 평균 887자)
#   - external_math_10000 + hybrid_verbose_distill 병합
#   - system 프롬프트를 baseline.py SYSTEM_PROMPTS["default"]로 통일 (12,687건 교체)
# 강도: rank32/all, LR 1e-5, 2 epoch
#
# epoch 1 지점(약 847 step) 체크포인트를 남겨 두 데이터 포인트를 한 번에 얻는다.
set -u
cd /workspace/DLC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

NAME=unified13k_r32_all_lr1e5_e2
DATA=data/processed/unified_prompt_13k.jsonl
HOLDOUT=data/holdout/official_holdout_464_clean.csv

echo "[$(date +%H:%M:%S)] TRAIN-START ${NAME}"
python3 scripts/train_qlora.py \
  --data "${DATA}" \
  --output-dir "checkpoints/${NAME}" \
  --learning-rate 1e-5 \
  --lora-rank 32 \
  --target-modules all \
  --epochs 2 \
  --batch-size 8 --gradient-accumulation 2 \
  --no-gradient-checkpointing \
  --save-steps 847 --save-total-limit 4 \
  --max-minutes 180 --seed 2026 > "logs/${NAME}.train.log" 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "[$(date +%H:%M:%S)] TRAIN-FAILED ${NAME} rc=${rc}"
  tail -25 "logs/${NAME}.train.log"
  exit 1
fi
echo "[$(date +%H:%M:%S)] TRAIN-DONE ${NAME}"

gate() {
  local tag="$1" adapter="$2"
  local out="outputs/${tag}_gate120.jsonl"
  echo "[$(date +%H:%M:%S)] GATE-START ${tag}"
  python3 scripts/baseline.py \
    --adapter-path "${adapter}" \
    --input "${HOLDOUT}" --limit 120 \
    --output "${out}" \
    --max-new-tokens 1024 --batch-size 60 > "logs/${tag}.gate.log" 2>&1
  if [ $? -ne 0 ]; then echo "[$(date +%H:%M:%S)] GATE-FAILED ${tag}"; tail -20 "logs/${tag}.gate.log"; return 1; fi
  python3 scripts/report_absorption.py "${out}" "${tag}"
}

# epoch 1 체크포인트가 남아 있으면 먼저 채점한다.
for ck in checkpoints/${NAME}/checkpoint-847; do
  [ -d "$ck" ] && gate "${NAME}_ep1" "$ck"
done
gate "${NAME}_ep2" "checkpoints/${NAME}/final_adapter"

echo "[$(date +%H:%M:%S)] BIGRUN-DONE"
