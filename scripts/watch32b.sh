#!/usr/bin/env bash
# 32B 전량 생성 라이브 모니터.  bash scripts/watch32b.sh
R=/workspace/DLC
TOTAL=5427
BASE=1625   # 스모크가 쓴 요청 수 (전량 시작 시점의 200 OK 누적)
while true; do
  done_=$(( $(grep -c "200 OK" $R/logs/teacher32b_server.log) - BASE ))
  [ $done_ -lt 0 ] && done_=0
  eng=$(grep "Avg generation throughput" $R/logs/teacher32b_server.log | tail -1 \
        | grep -oE "generation throughput: [0-9.]+ tokens/s, Running: [0-9]+")
  gpu=$(nvidia-smi --query-gpu=utilization.gpu,power.draw --format=csv,noheader)
  pct=$(( done_ * 100 / TOTAL ))
  bar=$(printf '█%.0s' $(seq 1 $((pct/4+1))))
  printf '\r\033[K[%s] %s/%s (%d%%) %-26s GPU %s | %s' \
    "$(TZ=Asia/Seoul date +%H:%M:%S)" "$done_" "$TOTAL" "$pct" "$bar" "$gpu" "$eng"
  if grep -q "전량 완료\|끝" $R/logs/teacher_full.log 2>/dev/null; then
    echo; echo "완료!"; tail -25 $R/logs/teacher_full.log; break
  fi
  sleep 5
done
