#!/usr/bin/env bash
# TIR 스모크 현황 한 눈에. `watch -n5 bash scripts/tir_status.sh` 로 쓰면 실시간.
cd /workspace/DLC
echo "=== 단계 ==="
cat logs/tir_main.log 2>/dev/null

echo
echo "=== 진행률 ==="
for n in unsolved68 gate120 remaining289; do
  if [ -f "logs/tir_${n}.log" ]; then
    printf "%-12s %s\n" "$n" "$(tr '\r' '\n' < logs/tir_${n}.log | grep -o 'tir: *[0-9]*%[^|]*|[^|]*' | tail -1)"
  fi
done

echo
echo "=== 채점 ==="
python3 - <<'PY'
import json, os
for name, total in (("unsolved68", 68), ("gate120", 120), ("remaining289", 289)):
    path = f"outputs/tir_{name}_hybrid3145.jsonl"
    if not os.path.exists(path):
        continue
    rows = [json.loads(l) for l in open(path)]
    if not rows:
        continue
    correct = sum(1 for r in rows if r["correct"])
    executed = sum(1 for r in rows if r["code_executed"])
    ok = sum(1 for r in rows if r.get("exec_status") == "ok")
    err = sum(1 for r in rows if r.get("exec_status") == "error")
    to = sum(1 for r in rows if r.get("exec_status") == "timeout")
    print(f"{name:11s} {len(rows):3d}/{total}  정답 {correct:3d}  "
          f"코드생성 {executed:3d} (실행성공 {ok} / 에러 {err} / 타임아웃 {to})")
PY

echo
echo "=== GPU ==="
nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader
date +%H:%M:%S
