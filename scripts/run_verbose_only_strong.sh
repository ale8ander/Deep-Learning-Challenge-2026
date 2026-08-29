#!/usr/bin/env bash
# unified13k(external 74%) 강한 학습이 흡수는 성공했으나 점수는 95→74→66으로 붕괴했다.
# 흡수한 스타일이 MATH/Numina 교과서체(간결·검산 없음)였기 때문이다.
#
# 이 실험은 변수를 하나만 바꾼다: **같은 강도, 데이터만 verbose 상세 CoT 863건으로 교체.**
# verbose는 유일하게 검산·케이스 전수·역추적을 보여주는 데이터이며(평균 1,473자),
# system 프롬프트도 이미 추론용으로 맞춰져 있다.
#
# 판정: 흡수 신호(**Approach:** 마커)가 유지되면서 점수가 74보다 높으면
#       "강도는 맞고 데이터가 문제였다"가 확정되고, 5,427개 확대($2.3)가 정당화된다.
set -u
cd /workspace/DLC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

DATA=data/processed/verbose_only_863.jsonl
HOLDOUT=data/holdout/official_holdout_464_clean.csv

run() {
  local name="$1" epochs="$2"
  echo "[$(date +%H:%M:%S)] TRAIN-START ${name} (verbose 863, rank32/all, lr1e-5, ep=${epochs})"
  python3 scripts/train_qlora.py \
    --data "${DATA}" \
    --output-dir "checkpoints/${name}" \
    --learning-rate 1e-5 --lora-rank 32 --target-modules all \
    --epochs "${epochs}" \
    --batch-size 8 --gradient-accumulation 2 \
    --no-gradient-checkpointing \
    --max-minutes 60 --seed 2026 > "logs/${name}.train.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] TRAIN-FAILED ${name}"; tail -20 "logs/${name}.train.log"; return 1
  fi
  echo "[$(date +%H:%M:%S)] TRAIN-DONE ${name}"
  python3 scripts/baseline.py \
    --adapter-path "checkpoints/${name}/final_adapter" \
    --input "${HOLDOUT}" --limit 120 \
    --output "outputs/${name}_gate120.jsonl" \
    --max-new-tokens 1024 --batch-size 60 > "logs/${name}.gate.log" 2>&1
  if [ $? -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] GATE-FAILED ${name}"; tail -20 "logs/${name}.gate.log"; return 1
  fi
  python3 scripts/report_absorption.py "outputs/${name}_gate120.jsonl" "${name}"
}

# 863건은 작으므로 epoch을 늘려 노출량을 확보한다. 3ep = 약 162 step.
run verbose863_r32_all_lr1e5_e3 3
# 중간 강도 대조군: 같은 데이터, rank16/qv, LR 5e-6
echo "[$(date +%H:%M:%S)] TRAIN-START verbose863_r16_qv_lr5e6_e3"
python3 scripts/train_qlora.py \
  --data "${DATA}" \
  --output-dir "checkpoints/verbose863_r16_qv_lr5e6_e3" \
  --learning-rate 5e-6 --lora-rank 16 --target-modules qv \
  --epochs 3 --batch-size 8 --gradient-accumulation 2 \
  --no-gradient-checkpointing \
  --max-minutes 60 --seed 2026 > "logs/verbose863_r16_qv_lr5e6_e3.train.log" 2>&1 \
  && python3 scripts/baseline.py \
    --adapter-path "checkpoints/verbose863_r16_qv_lr5e6_e3/final_adapter" \
    --input "${HOLDOUT}" --limit 120 \
    --output "outputs/verbose863_r16_qv_lr5e6_e3_gate120.jsonl" \
    --max-new-tokens 1024 --batch-size 60 > "logs/verbose863_r16_qv_lr5e6_e3.gate.log" 2>&1 \
  && python3 scripts/report_absorption.py "outputs/verbose863_r16_qv_lr5e6_e3_gate120.jsonl" "verbose863_r16_qv_lr5e6_e3"

echo "[$(date +%H:%M:%S)] VERBOSE863-DONE"
