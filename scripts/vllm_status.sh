#!/usr/bin/env bash
# vLLM 실행 상태판. 로그가 조용한 import 구간에도 진행을 보여준다.
#   watch -n5 bash scripts/vllm_status.sh
cd /workspace/DLC

LOG=${LOG:-logs/tir_sc8_holdout.log}
OUT=${OUT:-outputs/tir_sc8_holdout464_vote3.jsonl}
TOTAL=${TOTAL:-87}

echo "$(TZ=Asia/Seoul date '+%H:%M:%S KST')"
echo

PID=$(pgrep -f "[t]ir_inference_vllm" | head -1)
if [ -z "$PID" ]; then
  echo "프로세스: 종료됨"
  if grep -qE "^Traceback|^[A-Za-z]*Error:|RuntimeError|FileNotFoundError" "$LOG" 2>/dev/null; then
    echo "  ⚠ 에러 발견:"
    grep -E "^Traceback|^[A-Za-z]*Error:|RuntimeError|FileNotFoundError" "$LOG" | tail -3 | sed 's/^/    /'
  fi
else
  echo "프로세스: PID $PID  경과 $(ps -o etime= -p "$PID" | tr -d ' ')  CPU $(ps -o time= -p "$PID" | tr -d ' ')"
  # import 구간에는 지금 읽고 있는 파일이 유일한 진행 신호다
  READING=$(ls -l /proc/$PID/fd 2>/dev/null | grep -oE '/workspace/venv-vllm/[^ ]*|/workspace/models/[^ ]*' | tail -1)
  CHILD=$(pgrep -P "$PID" | head -1)
  if [ -n "$CHILD" ]; then
    CREADING=$(ls -l /proc/$CHILD/fd 2>/dev/null | grep -oE '/workspace/venv-vllm/[^ ]*|/workspace/models/[^ ]*' | tail -1)
    [ -n "$CREADING" ] && READING="$CREADING"
  fi
  [ -n "$READING" ] && echo "  읽는 중: ...${READING: -60}"
fi

echo
GPU=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader)
if [ -z "$GPU" ]; then
  echo "GPU: (아직 모델 안 올라감 — 초기화 중)"
else
  echo "GPU: $GPU"
fi

echo
if [ -f "$OUT" ]; then DONE=$(wc -l < "$OUT"); else DONE=0; fi
echo "출력: ${DONE}/${TOTAL}"
if [ "$DONE" -gt 0 ]; then
  python3 -c "
import json
rows=[json.loads(l) for l in open('$OUT')]
ok=sum(1 for r in rows if r['correct'])
adopted=sum(1 for r in rows if r['prediction'] is not None)
print(f'  채택 {adopted}  정답 {ok}')
" 2>/dev/null
fi

echo
echo "--- 로그 (${LOG}, $(stat -c %s "$LOG" 2>/dev/null || echo 0)바이트) ---"
if [ -s "$LOG" ]; then
  # vLLM 설정 덤프는 한 줄이 수천 자라 그냥 tail하면 화면이 도배된다. 헤더만 추린다.
  grep -oE "\[[a-z_0-9]+\.py:[0-9]+\].{0,65}" "$LOG" | tail -4
  tr '\r' '\n' < "$LOG" | grep -E "Processed prompts|Adding requests" | tail -2 | cut -c1-100
else
  echo "  (아직 출력 없음 — 네트워크에서 vLLM import 중, 약 6분 소요)"
fi
