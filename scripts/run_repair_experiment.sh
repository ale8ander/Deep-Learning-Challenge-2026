#!/usr/bin/env bash
# 코드 리페어 / adaptive 단일변수 실험.
#
# 대조군: outputs/bench_5090_vote3.jsonl
#   같은 pod, 같은 엔진(vLLM+네이티브 샘플러), 같은 시드(20260828), 같은 87문제.
#   correct 15/87, adopted 27, exec_error 51, exec_timeout 11.
#   A100 시절 산출물과 섞으면 안 되므로(샘플러 교체로 6.2% 발산) 이걸 기준으로 쓴다.
#
# 실험군:
#   1) --repair-rounds 1  : 실행 실패 샘플에 코드 수정 기회 1회
#   2) --repair-rounds 1 --adaptive-extra 8 : 위 + plurality 미확정 문제에만 8샘플 추가
set -u
cd /workspace/DLC
V=/workspace/venv-vllm/bin/python

echo "[$(date -u +%H:%M:%S)] 선행 생성 종료 대기"
while pgrep -f "tir_inference_client" >/dev/null; do sleep 15; done
echo "[$(date -u +%H:%M:%S)] 선행 종료 확인"
curl -s -m 5 localhost:8000/v1/models >/dev/null || { echo "서버 죽음"; exit 1; }

echo "[$(date -u +%H:%M:%S)] === 실험1: repair-rounds 1 ==="
$V scripts/tir_repair_client.py \
  --input data/holdout/holdout464_vote3.csv \
  --output outputs/tir_repair1_holdout464_vote3.jsonl \
  --model hybrid3145 --num-samples 8 --repair-rounds 1 \
  --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed 20260828

echo "[$(date -u +%H:%M:%S)] === 실험2: repair 1 + adaptive +8 ==="
$V scripts/tir_repair_client.py \
  --input data/holdout/holdout464_vote3.csv \
  --output outputs/tir_repair1_adapt8_holdout464_vote3.jsonl \
  --model hybrid3145 --num-samples 8 --repair-rounds 1 \
  --adaptive-extra 8 --adaptive-min-count 2 \
  --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed 20260828

echo "[$(date -u +%H:%M:%S)] 실험 완료"
