#!/usr/bin/env bash
# verifier 4차 홀드아웃 판정 — 자기증류와 **독립**으로 돈다.
#
# dev 73.79% 통과는 필요조건일 뿐이다. 이 프로젝트는 홀드아웃→Public 3연패,
# 오라클→실채택 전환 실패 6회 전력이 있다. 진짜 질문:
#   현행 배포 규칙(코드검증 plurality + min-count)에 verifier 재순위를 얹으면
#   **챔피언 대비** 순이득이 늘어나는가?
#
# verifier 는 대체가 아니라 재순위로만 쓴다 — 코드검증 통과 후보 중 P(A) 최고.
# dev 선택 시뮬에서도 이 조합(205)이 verifier 단독(199)보다 높았다.
set -uo pipefail
ROOT=/workspace/DLC
VPY=/workspace/venv-vllm/bin/python
PY=python3
LOGS=$ROOT/logs
OUT=$ROOT/outputs
VF=$ROOT/checkpoints/verifier_tir_r8_qv_lr2e5_e1/final_adapter
HOLDOUT=$ROOT/data/holdout/holdout464_vote3.csv

say() { echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] $*"; }
die() { echo "[중단] $*" >&2; exit 1; }
server_up() { curl -s -m 3 -o /dev/null http://localhost:8000/v1/models; }

# ── 1) 서버 기동 ──
say "=== 1) vLLM 서버 기동 (12~13분) ==="
if ! server_up; then
  VLLM_GPU_FRAC=0.90 setsid nohup bash "$ROOT/scripts/vllm_server.sh" \
      > "$LOGS/vllm.log" 2>&1 < /dev/null &
  for _ in $(seq 1 120); do sleep 15; server_up && break; done
  server_up || die "서버 기동 타임아웃"
fi
say "서버 준비 완료"

# ── 2) 홀드아웃87 궤적 생성 (hybrid3145, 8샘플) ──
# 기존 tir_repair1_holdout464_vote3.jsonl 에는 응답 원문이 없어서 verifier 입력을 못 만든다.
# --dump-trajectories 로 다시 뽑는다. 배포와 동일한 설정(리페어1 + 미생성재시도1).
say "=== 2) 홀드아웃87 궤적 생성 ==="
TRAJ=$OUT/verifier_holdout87_hybrid3145_traj.jsonl
if [ ! -f "$TRAJ" ]; then
  $VPY "$ROOT/scripts/tir_repair_client.py" \
    --input "$HOLDOUT" --output "$OUT/verifier_holdout87_hybrid3145.jsonl" \
    --dump-trajectories "$TRAJ" \
    --model hybrid3145 --num-samples 8 --repair-rounds 1 --nocode-retries 1 \
    --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed 20260906 \
    > "$LOGS/verifier_holdout87_gen.log" 2>&1 || { tail -20 "$LOGS/verifier_holdout87_gen.log"; die "생성 실패"; }
fi
tail -2 "$LOGS/verifier_holdout87_gen.log"
say "궤적 $(wc -l < "$TRAJ")개"

# ── 3) 챔피언 홀드아웃 등가물 ──
say "=== 3) 챔피언 등가물 생성 ==="
$PY "$ROOT/scripts/compose_holdout464.py" > "$LOGS/compose_holdout464.log" 2>&1 \
  || { tail -20 "$LOGS/compose_holdout464.log"; die "compose 실패"; }
sed -n '1,25p' "$LOGS/compose_holdout464.log"

# ── 4) verifier 판정 (서버 내리고 GPU 확보) ──
say "=== 4) verifier 홀드아웃 판정 ==="
pids=$(pgrep -f "vllm.entrypoints.openai.api_server" || true)
[ -n "$pids" ] && { say "서버 종료 (PID: $pids)"; kill $pids; sleep 20; }
$PY "$ROOT/scripts/apply_verifier_holdout.py" \
  --traj "$TRAJ" --adapter-path "$VF" \
  --champion "$OUT/champion_holdout464_equivalent.jsonl" \
  --min-count 2 --batch-size 16 \
  --output "$OUT/verifier_holdout87_verdict.json" \
  2>&1 | tee "$LOGS/verifier_holdout_verdict.log"

say "=== 완료 ==="
