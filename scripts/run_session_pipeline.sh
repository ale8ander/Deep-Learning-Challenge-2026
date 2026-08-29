#!/usr/bin/env bash
# 벤치마크 뒤에 이어붙는 작업들. GPU 유휴를 막으려 체이닝한다.
# (CONTEXT 교훈: 체이닝을 걸었으면 몇 분 뒤 nvidia-smi로 실제로 도는지 확인할 것)
set -u
cd /workspace/DLC
V=/workspace/venv-vllm/bin/python

echo "[$(date -u +%H:%M:%S)] 벤치마크 완료 대기"
until [ -f outputs/bench_5090_vote3.jsonl ]; do
  pgrep -f "[a]pi_server" >/dev/null || { echo "서버 죽음"; exit 1; }
  sleep 15
done
echo "[$(date -u +%H:%M:%S)] 벤치마크 완료 확인"

# --- 1) 역검증 스모크: 표수<=3 홀드아웃 87문제, 5계보 후보 상위 4개 ---
echo "[$(date -u +%H:%M:%S)] 역검증 시작"
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
echo "[$(date -u +%H:%M:%S)] 역검증 완료"

# --- 2) 빠진 voter 2종 holdout464 (챔피언을 464에서 직접 채점하기 위해) ---
for spec in "hybrid_3145_verify:hybrid3145:verify"; do
  name="${spec%%:*}"; rest="${spec#*:}"; model="${rest%%:*}"; style="${rest#*:}"
  out="outputs/${name}_holdout464_retry2048.jsonl"
  [ -f "$out" ] && { echo "SKIP $name"; continue; }
  echo "[$(date -u +%H:%M:%S)] voter $name"
  $V scripts/gen_client.py --input data/holdout/official_holdout_464_clean.csv \
    --output "$out" --model "$model" --prompt-style "$style"
done
echo "[$(date -u +%H:%M:%S)] 파이프라인 완료"
