#!/usr/bin/env bash
# 진행/ETA 상태판.  1회: bash scripts/eta.sh   갱신: bash scripts/eta.sh -w
R=/workspace/DLC
G='\033[32m'; Y='\033[33m'; RD='\033[31m'; D='\033[2m'; B='\033[1m'; N='\033[0m'

secs_since(){ # $1=파일 -> 수정 후 경과초
  [ -f "$1" ] || { echo -1; return; }
  echo $(( $(date +%s) - $(stat -c %Y "$1") ))
}
hms(){ printf '%d분 %02d초' $(( $1/60 )) $(( $1%60 )); }

board(){
clear 2>/dev/null
echo -e "${B}════════ $(TZ=Asia/Seoul date '+%m/%d %H:%M:%S') KST ════════${N}"
gpu=$(nvidia-smi --query-gpu=memory.used,memory.total,utilization.gpu --format=csv,noheader)
echo -e "GPU  ${gpu}"
echo

# ── ① R1 증류 판정 (포트 8000) ────────────────────────────────
echo -e "${B}① R1 long-CoT 증류 판정${N}  ${D}(P2)${N}"
if curl -s -m 2 http://localhost:8000/v1/models 2>/dev/null | grep -q hybrid3145; then
  echo -e "   3B 서버      ${G}준비완료${N}"
else
  st=$(grep -c . $R/logs/vllm.log 2>/dev/null || echo 0)
  echo -e "   3B 서버      ${Y}로딩 중${N} ${D}(로그 ${st}행)${N}"
  tail -1 $R/logs/vllm.log 2>/dev/null | sed 's/^/     /' | cut -c1-100
fi
for M in r1distill hybrid3145; do
  f=$R/outputs/r1distill_gen_${M}_holdout87.jsonl
  if [ -f "$f" ]; then
    n=$(wc -l < "$f")
    if [ "$n" -ge 87 ]; then
      echo -e "   gen ${M}  ${G}완료 ${n}/87${N}"
    else
      age=$(secs_since "$f")
      # 남은 문제수 x 지금까지의 문제당 평균초
      echo -e "   gen ${M}  ${Y}${n}/87${N} ${D}(최근 갱신 $(hms $age) 전)${N}"
    fi
  else
    echo -e "   gen ${M}  ${D}대기${N}"
  fi
done
if grep -q "판정" $R/logs/r1_eval.log 2>/dev/null; then
  echo -e "   ${G}▶ 판정 출력됨 — bash scripts/eta.sh -r 로 확인${N}"
fi
echo

# ── ② teacher 32B 스모크 (포트 8010) ──────────────────────────
echo -e "${B}② teacher 32B 스모크${N}  ${D}(P3, 1500문제 x N=2)${N}"
cache=$(du -sm /root/.cache/huggingface 2>/dev/null | cut -f1)
cache=${cache:-0}
if curl -s -m 2 http://localhost:8010/v1/models 2>/dev/null | grep -q teacher32b; then
  echo -e "   32B 서버     ${G}준비완료${N} ${D}(캐시 ${cache}MB)${N}"
elif [ "$cache" -gt 100 ]; then
  pct=$(( cache * 100 / 19000 ))
  echo -e "   32B 다운로드 ${Y}${cache}MB / ~19000MB (${pct}%)${N}"
else
  echo -e "   32B 서버     ${Y}vLLM import 중${N} ${D}(~5분, 캐시 ${cache}MB)${N}"
fi
f=$R/outputs/teacher32b_raw_smoke.jsonl
if [ -f "$f" ]; then
  n=$(wc -l < "$f"); age=$(secs_since "$f")
  if [ "$n" -ge 1500 ]; then
    echo -e "   생성         ${G}완료 ${n}/1500${N}"
  else
    echo -e "   생성         ${Y}${n}/1500${N} ${D}(최근 갱신 $(hms $age) 전)${N}"
  fi
else
  echo -e "   생성         ${D}대기${N}"
fi
tail -1 $R/logs/teacher_smoke.log 2>/dev/null | sed 's/^/   /' | cut -c1-110
echo


# ── ③ P1 표수4~5 스윕 ─────────────────────────────────────────
echo -e "${B}③ P1 표수4~5 전면적용 스윕${N}  ${D}(제출본 직결 레버)${N}"
for T in a100 r1 nc; do
  f=$R/outputs/tir_v45_${T}_holdout_vote45.jsonl
  if [ -f "$f" ]; then
    n=$(wc -l < "$f")
    if [ "$n" -ge 69 ]; then echo -e "   풀 ${T}\t${G}완료 ${n}/69${N}"
    else echo -e "   풀 ${T}\t${Y}${n}/69${N} ${D}($(hms $(secs_since "$f")) 전 갱신)${N}"; fi
  else
    prog=$(grep -o "[0-9]*/69" $R/logs/p1_${T}.log 2>/dev/null | tail -1)
    echo -e "   풀 ${T}\t${Y}생성 중${N} ${D}${prog}${N}"
  fi
done
tail -1 $R/logs/p1_vote45.log 2>/dev/null | sed 's/^/   /' | cut -c1-110
echo

# ── 실행 중 프로세스 ──────────────────────────────────────────
echo -e "${D}실행 중:${N}"
ps aux | grep -E "[a]pi_server|[g]en_client|[g]en_teacher|[t]ir_repair" \
  | awk '{printf "   pid %-7s %s\n", $2, $13}' | cut -c1-90
}

if [ "${1:-}" = "-r" ]; then tail -40 $R/logs/r1_eval.log; exit 0; fi
if [ "${1:-}" = "-p" ]; then tail -60 $R/logs/p1_vote45.log; exit 0; fi
if [ "${1:-}" = "-w" ]; then while true; do board; sleep 5; done; else board; fi
