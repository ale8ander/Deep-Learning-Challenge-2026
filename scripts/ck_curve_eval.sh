#!/usr/bin/env bash
# GRPO 체크포인트 곡선 자동 평가 — 50 step 마다 저장되는 체크포인트를
# 상주 서버에 핫로드해 holdout464 greedy 로 채점하고 요약을 누적한다.
# ck100=363, ck150=365 는 기평가. 평가 후 어댑터는 unload 해 슬롯을 비운다.
set -u
R=/workspace/DLC
V=/workspace/venv-vllm/bin/python
CKD=$R/checkpoints/grpo_3145_scaleup_r8_qv_lr2e6_steps800_g8
SUM=$R/outputs/grpo_scaleup_curve_summary.txt
say(){ echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] $*"; }
up(){ curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q hybrid3145; }
cd $R
touch $SUM

for N in $(seq 200 50 800); do
  # 체크포인트 대기: 파일이 생기거나, 학습이 죽고 더 안 나오면 종료
  while [ ! -f "$CKD/checkpoint-$N/adapter_config.json" ]; do
    if ! kill -0 7894 2>/dev/null; then
      say "학습 종료 감지 — checkpoint-$N 없음, 곡선 평가 마감"
      # 최종 어댑터가 있으면 그것까지 평가
      if [ -f "$CKD/final_adapter/adapter_config.json" ] && ! grep -q final $SUM; then
        up || exit 0
        curl -s -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \
          -d "{\"lora_name\":\"ckfinal\",\"lora_path\":\"$CKD/final_adapter\"}" >/dev/null
        OUT=$($V scripts/gen_client.py --input data/holdout/official_holdout_464_clean.csv \
          --output outputs/grpo_scaleup_ckfinal_holdout464.jsonl --model ckfinal \
          --request-workers 48 2>&1 | tail -1)
        say "final: $OUT" | tee -a $SUM
      fi
      exit 0
    fi
    sleep 120
  done
  up || { say "서버 다운 — 평가 불가, checkpoint-$N 건너뜀 안 하고 대기"; until up; do sleep 60; done; }
  curl -s -X POST localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \
    -d "{\"lora_name\":\"ck$N\",\"lora_path\":\"$CKD/checkpoint-$N\"}" >/dev/null
  OUT=$($V scripts/gen_client.py --input data/holdout/official_holdout_464_clean.csv \
    --output outputs/grpo_scaleup_ck${N}_holdout464.jsonl --model ck$N \
    --request-workers 48 2>&1 | tail -1)
  say "ck$N: $OUT" | tee -a $SUM
  curl -s -X POST localhost:8000/v1/unload_lora_adapter -H 'Content-Type: application/json' \
    -d "{\"lora_name\":\"ck$N\"}" >/dev/null
done
say "곡선 평가 전체 완료"
