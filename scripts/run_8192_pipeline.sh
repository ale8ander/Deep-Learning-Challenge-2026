#!/usr/bin/env bash
# max_model_len 8192 서버에서 토큰 상한을 올려 게이트를 재생성한다.
#
# 근거: CONTEXT 11절에서 상한 768 -> 2048 로 올렸을 때 "코드 없음" 54개 중 50개(93%)가
# 지시 무시가 아니라 잘림이었고 +8 을 얻었다. 지금 상한 2048 에서도 게이트 샘플의
# 21.8%(491/2256)가 코드 없음이다. 같은 병목이 남아 있는지 먼저 재고, 그 다음 본 생성.
#
# 4096 컨텍스트에서는 max_new_tokens 를 2048 초과로 못 올렸다(리페어가 턴을 더하면 400).
# 8192 로 올렸으므로 라운드1 을 4096 까지 줄 수 있다.
#
# ⚠️ 순서 원칙: **싸고 정보가치 높은 것부터.**
#   1) 진단 40문제 (~5분)  — 잘림이 실제로 병목인가
#   2) 홀드아웃 87문제 (~10분) — 정답이 있으므로 이득 여부를 판정할 수 있다
#   3) 게이트 282문제 (~25분) — 제출용 생성. **2번이 통과했을 때만 의미가 있다**
# 오늘 이 순서를 두 번 어겨서 GPU 40분을 상한 낮은 실험에 썼다.
set -u
cd /workspace/DLC
V=/workspace/venv-vllm/bin/python

echo "[$(date -u +%H:%M:%S)] 서버 대기"
until curl -s -m 2 localhost:8000/v1/models >/dev/null 2>&1; do
  pgrep -f "api_server" >/dev/null || { echo "서버 죽음"; exit 1; }
  sleep 10
done
echo "[$(date -u +%H:%M:%S)] 서버 UP (max-model-len 8192)"

echo "[$(date -u +%H:%M:%S)] === 1) 진단: 상한 2048 ==="
$V scripts/diag_truncation.py --max-new-tokens 2048 --limit 40
echo "[$(date -u +%H:%M:%S)] === 1) 진단: 상한 4096 ==="
$V scripts/diag_truncation.py --max-new-tokens 4096 --limit 40

echo "[$(date -u +%H:%M:%S)] === 2) 홀드아웃 검증 87문제 (상한 4096 + 리페어) ==="
$V scripts/tir_repair_client.py \
  --input data/holdout/holdout464_vote3.csv \
  --output outputs/tir_repair_tok4096_holdout464_vote3.jsonl \
  --model hybrid3145 --num-samples 8 --repair-rounds 1 \
  --max-new-tokens 4096 --repair-max-new-tokens 2048 --final-max-new-tokens 1024 \
  --exec-timeout 60 --exec-workers 32 --request-workers 24 --seed 20260906

echo "[$(date -u +%H:%M:%S)] === 홀드아웃 완료. 판정 후 게이트 생성 여부 결정 ==="
echo "[$(date -u +%H:%M:%S)] === 3) 게이트 282문제 생성 (상한 4096 + 리페어) ==="
$V scripts/tir_repair_client.py \
  --input data/holdout/tir_831_gate282.csv \
  --output outputs/tir_repair_tok4096_831_gate282.jsonl \
  --model hybrid3145 --num-samples 8 --repair-rounds 1 \
  --max-new-tokens 4096 --repair-max-new-tokens 2048 --final-max-new-tokens 1024 \
  --exec-timeout 60 --exec-workers 32 --request-workers 24 --seed 20260906

echo "[$(date -u +%H:%M:%S)] 파이프라인 완료"
