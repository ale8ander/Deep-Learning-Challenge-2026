#!/usr/bin/env bash
# 오늘 밤 연쇄 전용 상태판.  bash /workspace/DLC/scripts/mon.sh   (-w 로 10초 갱신)
#
# watch.sh 는 GPU/서버 같은 '순간 상태'만 본다. 이 스크립트는 그 위에
# **연쇄가 어느 단계까지 왔는지**를 산출물 파일 존재로 판정해서 보여준다.
# 로그가 끝에 한 번만 찍히는 하네스라 진행률을 로그로는 알 수 없기 때문이다.
set -u
D=/workspace/DLC
G="\033[32m"; Y="\033[33m"; R="\033[31m"; B="\033[2m"; N="\033[0m"

step() { # $1=상태(done/run/wait/fail) $2=이름 $3=비고
  case "$1" in
    done) printf "  ${G}✓${N} %-34s ${B}%s${N}\n" "$2" "$3" ;;
    run)  printf "  ${Y}▶${N} %-34s ${Y}%s${N}\n" "$2" "$3" ;;
    fail) printf "  ${R}✗${N} %-34s ${R}%s${N}\n" "$2" "$3" ;;
    *)    printf "  ${B}·${N} ${B}%-34s %s${N}\n" "$2" "$3" ;;
  esac
}
alive() { pgrep -f "$1" >/dev/null 2>&1; }
rows()  { [ -f "$1" ] && wc -l < "$1" | tr -d ' ' || echo 0; }

show() {
  clear 2>/dev/null
  echo "════════ $(TZ=Asia/Seoul date '+%m/%d %H:%M:%S') KST ════════"

  # ── GPU / 서버 ──
  printf "GPU  "
  nvidia-smi --query-gpu=utilization.gpu,memory.used,memory.total,temperature.gpu \
    --format=csv,noheader 2>/dev/null | sed 's/^/     /' || echo "n/a"
  printf "vLLM "
  if curl -s -m 2 localhost:8000/v1/models >/dev/null 2>&1; then
    curl -s -m 2 localhost:8000/metrics 2>/dev/null | awk '
      /^vllm:num_requests_running/ {r=$2} /^vllm:num_requests_waiting\{/ {w=$2}
      END {printf "     UP  실행 %d / 대기 %d\n", r, w}' 2>/dev/null \
      || echo "     UP"
  elif alive "api_server"; then echo "     기동중 (12~13분 소요)"
  else echo "     DOWN"; fi

  # ── 1) NC 풀 ──
  echo; echo "① NC 풀 생성 (내일 제출 탄약)"
  for S in 20260904 20260905; do
    f=$D/outputs/tir_nc_831_gate282_seed${S}.jsonl
    if [ -f "$f" ]; then step done "seed $S" "$(rows "$f")/282행"
    elif alive "tir_repair_client.py .*--seed $S"; then step run "seed $S" "생성 중 (~16분)"
    else step wait "seed $S" "대기"; fi
  done

  # ── 2) 자기증류 ──
  echo; echo "② 자기증류 continuation"
  ck=$D/checkpoints/tir_selfdistill_r8qv_lr5e7_e1/final_adapter/adapter_config.json
  if [ -f "$ck" ]; then step done "학습" "어댑터 생성됨"
  elif alive "train_qlora.py .*tir_selfdistill"; then
    p=$(tail -c 400 $D/logs/selfdistill_train3.log 2>/dev/null | tr '\r' '\n' | grep -oE '[0-9]+/[0-9]+ \[[0-9:]+<[0-9:]+' | tail -1)
    step run "학습" "${p:-시작 중}"
  else step wait "학습" "NC 풀 완료 후 시작"; fi
  for M in selfdistill hybrid3145; do
    f=$D/outputs/selfdistill_eval_${M}_holdout87.jsonl
    if [ -f "$f" ]; then step done "평가 $M" "$(rows "$f")/87행"
    elif alive "selfdistill_eval_${M}_holdout87"; then step run "평가 $M" "추론 중"
    else step wait "평가 $M" "대기"; fi
  done
  [ -f $D/logs/selfdistill_gate.log ] && step done "게이트 판정" "완료" || step wait "게이트 판정" "대기"

  # ── 3) 확정 사실 ──
  echo; echo "③ 오늘 확정"
  step done "재현 compose" "656 바이트 일치"
  step done "verifier 4차" "홀드아웃 기각 (선택 병목 9연패)"
  printf "  ${B}· 최고 후보: A100+R1+NC1 24샘플 mc2 = 홀드아웃 +9 (현행 +5)${N}\n"

  echo; echo "실행 중인 프로세스"
  ps -eo pid,etime,cmd 2>/dev/null \
    | grep -E "[t]ir_repair_client|[t]rain_qlora|[r]un_nc_pools|[r]un_selfdistill" \
    | awk '{printf "  %-7s %-9s %s\n", $1, $2, substr($0, index($0,$3), 60)}'
  echo "  (없으면 전부 종료된 것)"
}

if [ "${1:-}" = "-w" ]; then while :; do show; sleep 10; done; else show; fi
