#!/usr/bin/env bash
# vLLM 상주 서버 + 계보 실행 상태판.  watch -n10 bash scripts/server_status.sh
cd /workspace/DLC
echo "$(TZ=Asia/Seoul date '+%H:%M:%S KST')"
echo
R=$(curl -s --max-time 3 http://localhost:8000/v1/models 2>/dev/null)
if [ -n "$R" ]; then
  echo "$R" | python3 -c "import json,sys;print('✅ 서버 준비됨 —', [m['id'] for m in json.load(sys.stdin)['data']])" 2>/dev/null
else
  P=$(pgrep -f "[a]pi_server" | head -1)
  if [ -n "$P" ]; then echo "⏳ 서버 기동 중 (경과 $(ps -o etime= -p "$P" | tr -d ' '))"
  else echo "❌ 서버 없음"; grep -iE "error|unrecognized" logs/vllm_server.log | tail -2; fi
fi
echo
echo "--- 계보 실행 ---"
cat logs/tir_lineages.log 2>/dev/null | tail -6
for M in grpo96 verbose; do
  F="outputs/tir_sc8_holdout464_${M}.jsonl"
  if [ -f "$F" ]; then printf "  %-9s %s/87\n" "$M" "$(wc -l < "$F")"; else printf "  %-9s 대기\n" "$M"; fi
done
echo
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
