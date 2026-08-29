#!/usr/bin/env bash
# 한 화면 상태판. `bash /workspace/DLC/scripts/watch.sh` 로 1회, `-w` 로 5초마다 갱신.
#
# 경로는 전부 절대경로다 — 셸 cwd 가 /workspace 로 리셋되는 환경이라
# 상대경로로 쓰면 조용히 실패한다 (CONTEXT 2026-08-28 운영 교훈).
set -u
D=/workspace/DLC

show() {
  echo "════════ $(date -u +%H:%M:%S) UTC ════════"

  echo "── GPU ──"
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader 2>/dev/null || echo "  nvidia-smi 실패"

  echo "── vLLM 서버 ──"
  if curl -s -m 2 localhost:8000/v1/models >/dev/null 2>&1; then
    curl -s -m 2 localhost:8000/metrics 2>/dev/null | awk '
      /^vllm:num_requests_running/ {r=$2}
      /^vllm:num_requests_waiting\{/ {w=$2}
      /^vllm:generation_tokens_total/ {g=$2}
      END {printf "  UP  실행중 %d  대기 %d  누적생성 %.2fM tok\n", r, w, g/1e6}'
  else
    pgrep -f "api_server" >/dev/null && echo "  기동중(아직 응답 없음)" || echo "  DOWN"
  fi

  echo "── 실행 중인 작업 ──"
  ps -eo pid,etime,cmd 2>/dev/null | grep -E "[t]ir_repair_client|[t]ir_inference_client|[g]en_client|[v]erify_candidates|[t]rain_" \
    | awk '{printf "  pid %-7s %-9s %s\n", $1, $2, substr($0, index($0,$3), 58)}' \
    || echo "  없음"
  ps -eo cmd 2>/dev/null | grep -qE "[t]ir_|[g]en_client|[v]erify_can" || echo "  (없음 — 유휴)"

  echo "── 최근 로그 ──"
  for f in tir_831_vote68 repair1 repair_exp pipeline2 vllm_5090b; do
    [ -f "$D/logs/$f.log" ] || continue
    line=$(tail -n 40 "$D/logs/$f.log" 2>/dev/null \
           | grep -E "correct=|code_found|repair1_|elapsed|===|완료|Error|Traceback" \
           | tail -1)
    [ -n "$line" ] && printf "  %-18s %s\n" "$f" "$(echo "$line" | cut -c1-100)"
  done

  echo "── 산출물 ──"
  for f in tir_sc8_831_vote68 tir_repair1_holdout464_vote3 tir_repair1_adapt8_holdout464_vote3; do
    p="$D/outputs/$f.jsonl"
    if [ -f "$p" ]; then
      printf "  ✅ %-38s %s줄  %s\n" "$f" "$(wc -l < "$p")" "$(date -r "$p" -u +%H:%M:%S)"
    else
      printf "  ⏳ %-38s 생성중\n" "$f"
    fi
  done
  echo
}

if [ "${1:-}" = "-w" ]; then
  while true; do clear; show; sleep 5; done
else
  show
fi
