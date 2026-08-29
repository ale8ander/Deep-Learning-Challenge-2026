#!/usr/bin/env bash
set -u
cd /workspace/DLC
V=/workspace/venv-vllm/bin/python
echo "[$(date -u +%H:%M:%S)] seed2 종료 대기"
while [ ! -f outputs/tir_repair1_831_gate282_seed2.jsonl ]; do
  pgrep -f "api_server" >/dev/null || { echo "서버 죽음"; exit 1; }
  sleep 20
done
echo "[$(date -u +%H:%M:%S)] === 831 adaptive (4번째 풀) ==="
$V scripts/tir_repair_client.py --input data/holdout/tir_831_gate282.csv \
  --output outputs/tir_repair1_adapt8_831_gate282.jsonl \
  --model hybrid3145 --num-samples 8 --repair-rounds 1 \
  --adaptive-extra 8 --adaptive-min-count 2 \
  --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed 20260903
echo "[$(date -u +%H:%M:%S)] 완료"
