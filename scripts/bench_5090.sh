#!/usr/bin/env bash
# 서버가 뜰 때까지 기다렸다가 A100과 동일한 워크로드를 돌려 속도를 비교한다.
# A100 기준선: 87문제 x 8샘플 x 2라운드 = 222.7초 (exec-workers 96, 128코어)
set -u
cd /workspace/DLC
echo "[$(date +%H:%M:%S)] 서버 대기 시작"
until curl -s -m 2 localhost:8000/v1/models >/dev/null 2>&1; do sleep 10; done
echo "[$(date +%H:%M:%S)] 서버 UP"
/workspace/venv-vllm/bin/python scripts/tir_inference_client.py \
  --input data/holdout/holdout464_vote3.csv \
  --output outputs/bench_5090_vote3.jsonl \
  --model hybrid3145 --num-samples 8 --exec-workers 32 --seed 20260828
echo "[$(date +%H:%M:%S)] 완료"
