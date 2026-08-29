#!/usr/bin/env bash
# 2026-08-28 야간 2차 연쇄 — verifier 판정과 내일 제출탄 준비를 한 번에.
#
# 순서 (GPU 를 한 번도 놀리지 않는 배치):
#   1. verifier 학습 종료 대기
#   2. verifier dev 채점 + 선택 시뮬레이션  (kill switch: dev acc < 68.7% 폐기)
#   3. 자기증류 continuation 재학습 — OOM 수정판 (batch 2x8, 5090 32GB 용)
#   4. vLLM 서버 재기동 + 자기증류 어댑터 핫로드
#   5. 자기증류 홀드아웃87 평가 (신규 vs 기준, 같은 시드)
#   6. NC3/NC4 풀 생성 (게이트282, 새 시드 2개) — 내일 풀 쌓기 제출탄
set -uo pipefail
ROOT=/workspace/DLC
VPY=/workspace/venv-vllm/bin/python
PY=python3
LOGS=$ROOT/logs
OUT=$ROOT/outputs
SD_CKPT=$ROOT/checkpoints/tir_selfdistill_r8qv_lr5e7_e1
VF_CKPT=$ROOT/checkpoints/verifier_tir_r8_qv_lr2e5_e1
HOLDOUT=$ROOT/data/holdout/holdout464_vote3.csv
GATE=$ROOT/data/holdout/tir_831_gate282.csv

say() { echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] $*"; }
die() { echo "[중단] $*" >&2; exit 1; }
server_up() { curl -s -m 3 -o /dev/null http://localhost:8000/v1/models; }

start_server() {
  server_up && { say "서버 이미 떠 있음"; return 0; }
  say "vLLM 서버 기동 (12~13분)"
  VLLM_GPU_FRAC=0.90 setsid nohup bash "$ROOT/scripts/vllm_server.sh" \
      > "$LOGS/vllm.log" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do sleep 15; server_up && { say "서버 준비 완료"; return 0; }; done
  die "서버 기동 타임아웃"
}

# ── 1) verifier 학습 종료 대기 ──
say "=== 1) verifier 학습 대기 ==="
while pgrep -f "[t]rain_qlora.py .*verifier_tir_train" > /dev/null; do sleep 30; done
[ -f "$VF_CKPT/final_adapter/adapter_config.json" ] || die "verifier 어댑터 없음 (학습 실패)"
grep -oE '"train_loss": [0-9.]+' "$LOGS/verifier_tir_train.log" | tail -1

# ── 2) verifier dev 채점 ──
say "=== 2) verifier dev 채점 + 선택 시뮬레이션 ==="
$PY "$ROOT/scripts/score_verifier_tir.py" \
  --dev "$ROOT/data/processed/verifier_tir_dev.jsonl" \
  --adapter-path "$VF_CKPT/final_adapter" \
  --batch-size 16 \
  --output "$OUT/verifier_tir_dev_result.json" \
  > "$LOGS/verifier_tir_score.log" 2>&1 || { tail -20 "$LOGS/verifier_tir_score.log"; die "채점 실패"; }
cat "$OUT/verifier_tir_dev_result.json"

# ── 3) 자기증류 재학습 (OOM 수정: batch 2x8 — A100 의 8x2 는 32GB 에서 죽는다) ──
say "=== 3) 자기증류 continuation 재학습 ==="
if [ ! -f "$SD_CKPT/final_adapter/adapter_config.json" ]; then
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True $PY "$ROOT/scripts/train_qlora.py" \
    --data "$ROOT/data/processed/tir_selfdistill.jsonl" \
    --init-adapter "$ROOT/checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter" \
    --output-dir "$SD_CKPT" \
    --learning-rate 5e-7 --epochs 1 \
    --batch-size 2 --gradient-accumulation 8 \
    --max-seq-length 2048 --max-minutes 90 \
    --group-by-length --save-steps 5000 --seed 20260902 \
    > "$LOGS/selfdistill_train2.log" 2>&1 || { tail -20 "$LOGS/selfdistill_train2.log"; die "자기증류 학습 실패"; }
fi
say "자기증류 어댑터 준비됨"

# ── 4) 서버 재기동 + 핫로드 ──
say "=== 4) 서버 재기동 ==="
start_server
curl -s -m 60 -X POST http://localhost:8000/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"selfdistill\",\"lora_path\":\"$SD_CKPT/final_adapter\"}" \
  -w " (load_lora http=%{http_code})\n"

# ── 5) 자기증류 홀드아웃87 평가 ──
say "=== 5) 자기증류 평가 (신규 vs 기준, seed 20260903) ==="
for M in selfdistill hybrid3145; do
  o="$OUT/selfdistill_eval_${M}_holdout87.jsonl"
  [ -f "$o" ] && { say "SKIP $M"; continue; }
  $VPY "$ROOT/scripts/tir_repair_client.py" \
    --input "$HOLDOUT" --output "$o" \
    --dump-trajectories "$OUT/selfdistill_eval_${M}_traj.jsonl" \
    --model "$M" --num-samples 8 --repair-rounds 1 --nocode-retries 1 \
    --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed 20260903 \
    > "$LOGS/selfdistill_eval_${M}.log" 2>&1 || { tail -20 "$LOGS/selfdistill_eval_${M}.log"; die "평가 $M 실패"; }
  tail -2 "$LOGS/selfdistill_eval_${M}.log"
done
$PY "$ROOT/scripts/report_selfdistill_gate.py" \
  --new "$OUT/selfdistill_eval_selfdistill_holdout87.jsonl" \
  --base "$OUT/selfdistill_eval_hybrid3145_holdout87.jsonl" \
  --new-log "$LOGS/selfdistill_eval_selfdistill.log" \
  --base-log "$LOGS/selfdistill_eval_hybrid3145.log" \
  2>&1 | tee "$LOGS/selfdistill_gate.log"

# ── 6) NC3/NC4 풀 — 내일 풀 쌓기 제출탄 (검증된 E-2 방향) ──
say "=== 6) NC3/NC4 풀 생성 (게이트282 x 2시드) ==="
for SEED in 20260904 20260905; do
  o="$OUT/tir_nc_831_gate282_seed${SEED}.jsonl"
  [ -f "$o" ] && { say "SKIP seed $SEED"; continue; }
  $VPY "$ROOT/scripts/tir_repair_client.py" \
    --input "$GATE" --output "$o" \
    --model hybrid3145 --num-samples 8 --repair-rounds 1 --nocode-retries 1 \
    --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed "$SEED" \
    > "$LOGS/nc_gate282_seed${SEED}.log" 2>&1 || { tail -10 "$LOGS/nc_gate282_seed${SEED}.log"; say "seed $SEED 실패 (계속)"; continue; }
  tail -2 "$LOGS/nc_gate282_seed${SEED}.log"
done

say "=== 야간 2차 연쇄 완료 ==="
