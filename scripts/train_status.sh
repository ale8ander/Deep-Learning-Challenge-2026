#!/usr/bin/env bash
# 학습 상태판.  watch -n10 bash scripts/train_status.sh
# 다른 학습을 볼 때:  LOG=logs/다른로그.log OUT=checkpoints/다른체크포인트 watch -n10 bash scripts/train_status.sh
cd /workspace/DLC

LOG=${LOG:-logs/train_numina_tir_fixed.log}
OUT=${OUT:-checkpoints/numina_tir3000_r8_qv_lr2e6_e1_fixed}

echo "$(TZ=Asia/Seoul date '+%H:%M:%S KST')"
echo

PID=$(pgrep -f "[t]rain_qlora.py" | head -1)
if [ -z "$PID" ]; then
  echo "학습: 종료됨"
  if [ -d "$OUT/final_adapter" ]; then
    echo "  ✅ final_adapter 생성됨"
    python3 -c "
import json
try:
    m=json.load(open('$OUT/training_metadata.json'))
    me=m.get('metrics',{})
    print(f\"  소요 {me.get('train_runtime',0):.0f}초  최종 train_loss {me.get('train_loss',0):.4f}  epoch {me.get('epoch',0)}\")
except Exception as e: pass
" 2>/dev/null
  elif grep -qE "^Traceback|Error" "$LOG" 2>/dev/null; then
    echo "  ⚠ 에러:"
    grep -E "^[A-Za-z]*Error|ValueError|RuntimeError" "$LOG" | tail -2 | sed 's/^/    /'
  fi
else
  echo "학습: PID $PID  경과 $(ps -o etime= -p "$PID" | tr -d ' ')"
fi

echo
echo "--- 진행 ---"
tr '\r' '\n' < "$LOG" 2>/dev/null | grep -oE "[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+ \[[^]]*\]" | tail -1

echo
echo "--- loss 추이 (최근 6개 기록) ---"
tr '\r' '\n' < "$LOG" 2>/dev/null | grep -oE "'loss': [0-9.]+, 'grad_norm': [0-9.]+, 'learning_rate': [0-9.e-]+, 'epoch': [0-9.]+" \
  | tail -6 \
  | sed -E "s/'loss': ([0-9.]+), 'grad_norm': ([0-9.]+), 'learning_rate': ([0-9.e-]+), 'epoch': ([0-9.]+)/  loss \1   grad \2   lr \3   epoch \4/"

echo
echo "--- GPU ---"
nvidia-smi --query-gpu=utilization.gpu,memory.used --format=csv,noheader
