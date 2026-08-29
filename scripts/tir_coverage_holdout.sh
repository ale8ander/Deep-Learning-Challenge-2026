#!/usr/bin/env bash
# 표수 6~8 구간(TIR 미적용 영역)에 TIR self-consistency 를 처음 돌린다.
#
# 왜: 현재 TIR 은 831 중 282문제(34%)에만 적용된다. 표수 6~8 인 549문제는 손도 안 댔다.
# CONTEXT 21절이 계산으로 기각했지만 그건 min-count 2 기준이었고, 20절에서 이미
# "구간마다 threshold 를 따로 잡아야 한다"를 배웠다. 높은 min-count 에서 TIR 정밀도는
# 85~100% 로 오른다. 이 구간 baseline 은 90~99% 라 좁고 정확한 게이트만 통할 수 있다.
#
# 먼저 holdout464 의 해당 341문제로 판정하고, 양수일 때만 831 로 확대한다.
set -u
cd /workspace/DLC
V=/workspace/venv-vllm/bin/python

echo "[$(date -u +%H:%M:%S)] 역검증 종료 대기"
while pgrep -f "verify_candidates_client" >/dev/null; do sleep 15; done
echo "[$(date -u +%H:%M:%S)] 역검증 종료 확인"

curl -s -m 5 localhost:8000/v1/models >/dev/null || { echo "서버 죽음"; exit 1; }

echo "[$(date -u +%H:%M:%S)] === TIR SC 표수6~8 holdout464 (341문제) ==="
$V scripts/tir_inference_client.py \
  --input data/holdout/holdout464_tir_remaining.csv \
  --output outputs/tir_sc8_holdout464_remaining341.jsonl \
  --model hybrid3145 --num-samples 8 \
  --exec-timeout 60 --exec-workers 32 --request-workers 32 \
  --seed 20260828
echo "[$(date -u +%H:%M:%S)] 완료"
