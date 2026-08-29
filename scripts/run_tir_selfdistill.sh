#!/usr/bin/env bash
# TIR 자기증류 continuation 실험 — 수확 완료 후 전 단계를 자동으로 잇는다.
#
# ── 이 실험이 22절 'TIR SFT 재시도 금지'와 다른 점 ────────────────────────────
# 22절 실패 두 번은 (a) fresh base + (b) 외부 teacher(Numina) 코드 + (c) TIR-only 였다.
# 여기서는 세 변수를 전부 뒤집는다:
#   (a) hybrid_3145 에서 이어서 학습 (--init-adapter)
#   (b) 이 모델 자신이 코드검증 통과한 궤적만 사용
#   (c) hybrid_3145 원본을 1:1 replay 로 섞어 일반 능력 붕괴 방지
# CONTEXT 요약 23행이 이 조합만 "미검증, 다시 열어둔다"고 명시했다.
#
# ── 판정 원칙 (CONTEXT C절) ──────────────────────────────────────────────
#  * 점수보다 **흡수 지표를 먼저** 본다 (17절). no-code율/실행성공률/코드검증 오라클이
#    안 움직이면 학습이 아무것도 안 한 것이므로 즉시 중단한다.
#  * 홀드아웃87 은 gain/reg 표준오차가 ±3.5~3.9다. ±4 미만 차이를 실효과로 읽지 말 것.
#  * 채택 기준은 '단독 점수'가 아니라 **기존 풀에 더했을 때 실채택이 오르는가**이다.
#
# 사용: setsid nohup bash /workspace/DLC/scripts/run_tir_selfdistill.sh \
#         > /workspace/DLC/logs/selfdistill_chain.log 2>&1 &
set -uo pipefail

ROOT=/workspace/DLC
VPY=/workspace/venv-vllm/bin/python
PY=python3
LOGS=$ROOT/logs
OUT=$ROOT/outputs
TAG=tir_selfdistill_r8qv_lr5e7_e1
CKPT=$ROOT/checkpoints/$TAG
HOLDOUT=$ROOT/data/holdout/holdout464_vote3.csv     # 87문제, 정답 있음 (판정용)

say() { echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] $*"; }
die() { echo "[중단] $*" >&2; exit 1; }

server_up() { curl -s -m 3 -o /dev/null http://localhost:8000/v1/models; }

stop_server() {
  # ⚠️ pkill -f <패턴> 은 자기 명령줄이 그 문자열을 담고 있으면 자기 자신을 죽인다
  # (CONTEXT 14절에서 두 번 밟은 함정). PID 로만 잡는다.
  local pids
  pids=$(pgrep -f "vllm.entrypoints.openai.api_server" || true)
  [ -n "$pids" ] || return 0
  say "vLLM 서버 종료 (PID: $pids)"
  # shellcheck disable=SC2086
  kill $pids
  for _ in $(seq 1 40); do sleep 5; server_up || return 0; done
  say "정상 종료 실패, 강제 종료"; kill -9 $pids 2>/dev/null; sleep 10
}

start_server() {
  server_up && { say "서버 이미 떠 있음"; return 0; }
  say "vLLM 서버 기동 (12~13분)"
  VLLM_GPU_FRAC=0.90 setsid nohup bash "$ROOT/scripts/vllm_server.sh" \
      > "$LOGS/vllm.log" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do sleep 15; server_up && { say "서버 준비 완료"; return 0; }; done
  die "서버 기동 타임아웃"
}

# ── 0) 수확 완료 대기 ─────────────────────────────────────────────────────
say "=== 0) 궤적 수확 완료 대기 ==="
while pgrep -f "[t]ir_repair_client.py .*tir_distill_pool" > /dev/null; do sleep 60; done
[ -s "$OUT/tir_selfdistill_pool3000_traj.jsonl" ] || die "궤적 덤프가 비었다"
say "수확 완료: $(wc -l < "$OUT/tir_selfdistill_pool3000_traj.jsonl")궤적"
tail -2 "$LOGS/selfdistill_pool3000.log"

# ── 1) SFT 데이터 변환 ────────────────────────────────────────────────────
say "=== 1) SFT 데이터 변환 (검증된 정답 궤적 + replay 1:1) ==="
$PY "$ROOT/scripts/build_tir_selfdistill_sft.py" \
  --traj "$OUT/tir_selfdistill_pool3000_traj.jsonl" \
  --pool "$ROOT/data/processed/tir_distill_pool.csv" \
  --output "$ROOT/data/processed/tir_selfdistill.jsonl" \
  --max-per-problem 2 --replay-ratio 1.0 \
  > "$LOGS/selfdistill_build.log" 2>&1 || { tail -20 "$LOGS/selfdistill_build.log"; die "변환 실패"; }
grep -E '"(tir_samples|replay_samples|total|kept_rescued)"' "$LOGS/selfdistill_build.log"

# ── 2) 학습 (서버 내리고 GPU 단독) ────────────────────────────────────────
# 32GB 에서 서버(0.90)와 학습을 같이 띄우면 OOM 이다 (22절 부수 발견).
say "=== 2) continuation 학습 ==="
stop_server
sleep 10
$PY "$ROOT/scripts/train_qlora.py" \
  --data "$ROOT/data/processed/tir_selfdistill.jsonl" \
  --init-adapter "$ROOT/checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter" \
  --output-dir "$CKPT" \
  --learning-rate 5e-7 --epochs 1 \
  --batch-size 8 --gradient-accumulation 2 \
  --max-seq-length 2048 --max-minutes 90 \
  --group-by-length --seed 20260902 \
  > "$LOGS/selfdistill_train.log" 2>&1 || { tail -30 "$LOGS/selfdistill_train.log"; die "학습 실패"; }
[ -f "$CKPT/final_adapter/adapter_config.json" ] || die "어댑터가 안 나왔다"
say "학습 완료: $CKPT/final_adapter"
grep -E "train_loss|global_step" "$LOGS/selfdistill_train.log" | tail -3

# ── 3) 서버 재기동 + 새 어댑터 핫로드 ─────────────────────────────────────
say "=== 3) 서버 재기동 + 어댑터 등록 ==="
start_server
curl -s -m 60 -X POST http://localhost:8000/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"selfdistill\",\"lora_path\":\"$CKPT/final_adapter\"}" \
  -o "$LOGS/selfdistill_loadlora.log" -w "load_lora http=%{http_code}\n"
curl -s -m 10 http://localhost:8000/v1/models | \
  $VPY -c "import json,sys; print('등록 어댑터:', [m['id'] for m in json.load(sys.stdin)['data']])"

# ── 4) 홀드아웃 87 평가 (신규 vs 기준) ────────────────────────────────────
# 같은 시드/설정으로 두 모델을 나란히 돌린다. 변수는 어댑터 하나뿐이다.
say "=== 4) 홀드아웃87 평가 ==="
for M in selfdistill hybrid3145; do
  o="$OUT/selfdistill_eval_${M}_holdout87.jsonl"
  [ -f "$o" ] && { say "SKIP $M"; continue; }
  say "평가 $M"
  $VPY "$ROOT/scripts/tir_repair_client.py" \
    --input "$HOLDOUT" --output "$o" \
    --dump-trajectories "$OUT/selfdistill_eval_${M}_traj.jsonl" \
    --model "$M" --num-samples 8 --repair-rounds 1 --nocode-retries 1 \
    --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed 20260903 \
    > "$LOGS/selfdistill_eval_${M}.log" 2>&1 || { tail -20 "$LOGS/selfdistill_eval_${M}.log"; die "평가 $M 실패"; }
  tail -2 "$LOGS/selfdistill_eval_${M}.log"
done

# ── 5) 흡수 지표 비교 (점수보다 먼저 본다) ────────────────────────────────
say "=== 5) 흡수 지표 + 오라클 비교 ==="
$PY "$ROOT/scripts/report_selfdistill_gate.py" \
  --new "$OUT/selfdistill_eval_selfdistill_holdout87.jsonl" \
  --base "$OUT/selfdistill_eval_hybrid3145_holdout87.jsonl" \
  --new-log "$LOGS/selfdistill_eval_selfdistill.log" \
  --base-log "$LOGS/selfdistill_eval_hybrid3145.log" \
  2>&1 | tee "$LOGS/selfdistill_gate.log"

say "=== 연쇄 완료 ==="
