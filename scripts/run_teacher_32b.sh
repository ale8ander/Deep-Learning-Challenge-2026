#!/usr/bin/env bash
# Qwen2.5-32B-Instruct teacher 로 우리 실패 풀(5,427문제)의 풀이를 생성한다.
#
# ── 왜 32B-Instruct 이고 R1-Distill-32B 가 아닌가 ─────────────────
# 성능이 아니라 **토큰 예산** 때문이다. R1 류 long-CoT 는 3,000~10,000+ 토큰이라
# 우리 학습 상한(max_seq 3072)을 뚫는다. 자르면 핵심이 사라지고 늘리면 학습이 4배 느려진다.
# 32B-Instruct 의 간결한 풀이(500~1,500토큰)는 현 파이프라인에 **무수정으로** 들어간다.
#
# ── 왜 이게 지금까지의 외부 데이터와 다른가 ──────────────────────
# external_3000/10000 은 "출처 불명 공개 CoT"였다. 이번은
#   (a) 우리 대회 train 분포 그대로,
#   (b) 우리 모델이 **틀리는 문제만** 골라서,
#   (c) gold 정답 대조로 검증한 풀이
# 세 조건을 처음으로 동시에 만족한다.
#
# ── 규정 ────────────────────────────────────────────────────────
# 공개·무료 가중치(Apache-2.0)를 **학습 데이터 구축**에 쓰는 것 — 규칙 명시 허용.
# 추론 시에는 사용하지 않는다. 최종 제출 시 외부 데이터 목록에 명시할 것.
set -u
R=/workspace/DLC
V=/workspace/venv-vllm/bin/python
TEACHER=Qwen/Qwen2.5-32B-Instruct-AWQ
POOL=$R/data/processed/distill_target_pool.jsonl
OUT=$R/outputs/teacher32b_raw.jsonl
say(){ echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] $*"; }

VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)
[ "$VRAM" -lt 60000 ] && { say "VRAM ${VRAM}MB — 32B 는 80GB 급이 필요하다. 중단"; exit 1; }

[ -f "$OUT" ] && { say "이미 생성됨: $(wc -l < "$OUT")행 — 건너뜀"; exit 0; }

# 3B 서버(0.45)와 공존시키려면 teacher 는 별도 포트 + 남은 VRAM 으로 띄운다.
say "teacher 서버 기동: $TEACHER (다운로드 포함 10~20분)"
HF_HUB_ENABLE_HF_TRANSFER=1 setsid nohup $V -m vllm.entrypoints.openai.api_server \
  --model "$TEACHER" --served-model-name teacher32b \
  --port 8001 --gpu-memory-utilization 0.50 \
  --max-model-len 4096 --disable-log-requests \
  > $R/logs/teacher32b_server.log 2>&1 < /dev/null &

for _ in $(seq 1 160); do
  sleep 15
  curl -s -m 3 -o /dev/null http://localhost:8001/v1/models && break
done
curl -s -m 3 -o /dev/null http://localhost:8001/v1/models || {
  say "teacher 기동 실패"; tail -20 $R/logs/teacher32b_server.log; exit 1; }
say "teacher 준비 완료"

say "생성 시작: $(wc -l < "$POOL")문제 x N=4"
$V $R/scripts/gen_teacher_client.py \
  --input "$POOL" --output "$OUT" \
  --base-url http://localhost:8001/v1 --model teacher32b \
  --num-samples 4 --max-new-tokens 1536 --temperature 0.7 \
  --request-workers 64 --seed 20260930 \
  > $R/logs/teacher32b_gen.log 2>&1 || { say "생성 실패"; tail -20 $R/logs/teacher32b_gen.log; exit 1; }
tail -3 $R/logs/teacher32b_gen.log

say "teacher 서버 종료"
tp=$(pgrep -f "served-model-name teacher32b" || true)
[ -n "$tp" ] && kill $tp

say "=== 검증 통과 풀이만 SFT 로 변환 ==="
python3 $R/scripts/build_teacher_sft.py \
  --raw "$OUT" --pool "$POOL" \
  --output $R/data/processed/teacher32b_sft.jsonl \
  --max-per-problem 2 --max-seq 3072
say "완료 — 아침에 게이트 결과 보고 학습 여부 결정"
