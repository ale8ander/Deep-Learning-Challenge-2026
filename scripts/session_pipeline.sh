#!/usr/bin/env bash
# 세션 파이프라인. 순서는 기대값 순이다.
#  1) 벤치마크 — 5090 vs A100 실측 (A100 기준선 222.7초, 87문제 x 8샘플 x 2라운드)
#  2) voter 2종 holdout464 — 챔피언을 464에서 직접 채점하기 위해 필요.
#     지금까지 규칙 판정을 hybrid_3145 기준으로 재고 x0.5 로 환산해 왔는데,
#     그 환산이 틀려서 제출이 세 번 죽었다(N=16 -1, N=16단독 -1, risky게이트 -1).
#     이걸 끝내야 다음 후보를 "챔피언 대비 gain/regression"으로 잴 수 있다.
#  3) 역검증 — 이번 세션의 본 베팅. 후보를 표가 아니라 내용으로 고른다.
#
# 주의: 이 스크립트를 죽일 때 `pkill -f session_pipeline` 를 쓰면 같은 문자열을 담은
# 자기 명령까지 죽는다. PID 로 잡을 것 (CONTEXT 2026-08-27 운영 교훈).
set -u
cd /workspace/DLC
V=/workspace/venv-vllm/bin/python

echo "[$(date -u +%H:%M:%S)] 서버 대기"
until curl -s -m 2 localhost:8000/v1/models >/dev/null 2>&1; do
  pgrep -f "api_server" >/dev/null || { echo "서버 죽음"; exit 1; }
  sleep 10
done
echo "[$(date -u +%H:%M:%S)] 서버 UP"

echo "[$(date -u +%H:%M:%S)] === 1) 벤치마크 ==="
$V scripts/tir_inference_client.py --input data/holdout/holdout464_vote3.csv \
  --output outputs/bench_5090_vote3.jsonl --model hybrid3145 \
  --num-samples 8 --exec-workers 32 --seed 20260828

echo "[$(date -u +%H:%M:%S)] === 2) voter holdout464 ==="
for spec in "hybrid_3145_verify:hybrid3145:verify" "hybrid_4145:h4145:default"; do
  name="${spec%%:*}"; rest="${spec#*:}"; model="${rest%%:*}"; style="${rest#*:}"
  out="outputs/${name}_holdout464_retry2048.jsonl"
  if [ -f "$out" ]; then echo "SKIP $name"; continue; fi
  $V scripts/gen_client.py --input data/holdout/official_holdout_464_clean.csv \
    --output "$out" --model "$model" --prompt-style "$style"
done

echo "[$(date -u +%H:%M:%S)] === 3) 역검증 ==="
$V scripts/verify_candidates_client.py \
  --input data/holdout/holdout464_vote3.csv \
  --candidates outputs/tir_sc8_holdout464_vote3_to60.jsonl \
               outputs/tirc_hybrid3145_holdout464_vote3.jsonl \
               outputs/tirc_grpo96_holdout464_vote3.jsonl \
               outputs/tirc_tirsft_holdout464_vote3.jsonl \
               outputs/tirc_tirexec_holdout464_vote3.jsonl \
  --baseline outputs/self_consistency_confidence_n8_holdout464.jsonl \
  --output outputs/revverify_holdout464_vote3.jsonl \
  --model hybrid3145 --max-candidates 4 --num-samples 4 \
  --exec-workers 32 --request-workers 32

echo "[$(date -u +%H:%M:%S)] 파이프라인 완료"
