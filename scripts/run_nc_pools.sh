#!/usr/bin/env bash
# 내일 제출 탄약: NC 설정 새 시드 풀 2개 (게이트282, 각 ~16분).
# 검증된 방향이다 — 8->16->24 샘플이 매번 +1~5 였고 아직 수익 체감 전이다.
# ⚠️ 프로세스 종료는 반드시 PID 로. pkill -f 는 자기 명령줄을 매칭해 자신을 죽인다(CONTEXT 14절).
set -u
say(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*"; }
up(){ curl -s -m 3 -o /dev/null http://localhost:8000/v1/models; }

if ! up; then
  say "vLLM 서버 기동 (12~13분)"
  VLLM_GPU_FRAC=0.90 setsid nohup bash /workspace/DLC/scripts/vllm_server.sh \
    > /workspace/DLC/logs/vllm.log 2>&1 < /dev/null &
  for _ in $(seq 1 120); do sleep 15; up && break; done
fi
up || { say "서버 기동 실패"; exit 1; }
say "서버 준비 완료"

for SEED in 20260904 20260905; do
  o=/workspace/DLC/outputs/tir_nc_831_gate282_seed${SEED}.jsonl
  [ -f "$o" ] && { say "SKIP seed $SEED"; continue; }
  say "NC 풀 seed $SEED 생성 시작"
  /workspace/venv-vllm/bin/python /workspace/DLC/scripts/tir_repair_client.py \
    --input /workspace/DLC/data/holdout/tir_831_gate282.csv --output "$o" \
    --model hybrid3145 --num-samples 8 --repair-rounds 1 --nocode-retries 1 \
    --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed "$SEED" \
    > /workspace/DLC/logs/nc_gate282_seed${SEED}.log 2>&1 \
    || { say "seed $SEED 실패"; continue; }
  say "seed $SEED 완료"; tail -2 /workspace/DLC/logs/nc_gate282_seed${SEED}.log
done
say "NC 풀 생성 전부 완료"
