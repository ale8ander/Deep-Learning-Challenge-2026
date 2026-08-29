#!/usr/bin/env bash
# 빈 상태 -> 최종 제출본까지 한 번에 도는 재현 스크립트 (CONTEXT 부채 I-1).
#
# 왜 필요한가: 현행 656 제출본은 8일간 4개 pod 에서 쌓인 산출물 조각으로 만들어진다.
# 주최측 재현 검증에서 "어떻게 만들었는지" 보여줄 수 없으면 TIR 이 규정 회색지대인
# 상황에서 수상이 위험해진다. pod 가 갈려도 복구가 빨라지므로 우리한테도 이득이다.
#
# ─────────────────────────────────────────────────────────────────────────────
# 두 가지 모드가 있고, 목적이 다르다. 반드시 구분할 것.
#
#   compose : 기존 산출물(jsonl) -> 최종 CSV.  결정론적이라 **바이트 단위로 재현된다.**
#             GPU 불필요, ~1분. 제출본이 규칙대로 조립됐음을 증명하는 용도.
#
#   full    : 빈 상태 -> 새 샘플 생성 -> 최종 CSV.  샘플링이 확률적이라
#             **바이트 단위 재현은 원리적으로 불가능하다** (CONTEXT 8절: 같은 시드라도
#             엔진/샘플러가 다르면 다른 샘플이 나온다). 규칙과 절차가 재현되는지를 본다.
#
# 사용:
#   bash scripts/reproduce_all.sh compose              # 조립 검증 (기본값)
#   bash scripts/reproduce_all.sh full --smoke 20      # 전체 경로를 20문제로 스모크
#   bash scripts/reproduce_all.sh full                 # 전체 재생성 (~2시간, GPU 점유)
#   bash scripts/reproduce_all.sh full --from tir      # 중간 단계부터 재개
#
# 재개 안전: 각 단계는 산출물이 이미 있으면 건너뛴다. 다시 만들려면 파일을 지운다.
set -uo pipefail

cd /workspace/DLC
ROOT=/workspace/DLC
PY=python3
VPY=/workspace/venv-vllm/bin/python          # vLLM 클라이언트 전용 venv
TEST=data/deep_chal_math_leaderboard_filtered.csv
REPRO=outputs/repro
LOGS=logs/repro
mkdir -p "$REPRO" "$LOGS"

MODE="${1:-compose}"; shift || true
SMOKE=""; FROM="voters"
while [ $# -gt 0 ]; do
  case "$1" in
    --smoke) SMOKE="$2"; shift 2 ;;
    --from)  FROM="$2";  shift 2 ;;
    *) echo "알 수 없는 인자: $1"; exit 2 ;;
  esac
done

say() { echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] $*"; }
die() { echo "[실패] $*" >&2; exit 1; }

# 단계 순서. --from 으로 중간부터 재개할 때 쓴다.
STAGES=(voters sc tir merge)
stage_idx() { local i=0; for s in "${STAGES[@]}"; do [ "$s" = "$1" ] && { echo $i; return; }; i=$((i+1)); done; echo 99; }
FROM_IDX=$(stage_idx "$FROM")
skip_stage() { [ "$(stage_idx "$1")" -lt "$FROM_IDX" ]; }

# ─────────────────────────────────────────────────────────────────────────────
# Stage 0 — 사전 점검. 여기서 걸러야 2시간 뒤에 실패하지 않는다.
# ─────────────────────────────────────────────────────────────────────────────
say "=== Stage 0: 사전 점검 ==="

# 주최측 재현 환경 요건: Python>=3.10, PyTorch>=2.0, CUDA>=12.0
$PY - <<'PYEOF' || die "환경 요건 미달"
import sys
ok = True
v = sys.version_info
print(f"  Python {v.major}.{v.minor}.{v.micro}", end="")
if (v.major, v.minor) < (3, 10):
    print("  ✗ (>=3.10 필요)"); ok = False
else:
    print("  ✓")
try:
    import torch
    tv = tuple(int(x) for x in torch.__version__.split("+")[0].split(".")[:2])
    print(f"  PyTorch {torch.__version__}", end="")
    print("  ✓" if tv >= (2, 0) else "  ✗ (>=2.0 필요)")
    ok = ok and tv >= (2, 0)
    cu = torch.version.cuda
    print(f"  CUDA {cu}", end="")
    cv = tuple(int(x) for x in (cu or "0").split(".")[:2])
    print("  ✓" if cv >= (12, 0) else "  ✗ (>=12.0 필요)")
    ok = ok and cv >= (12, 0)
    print(f"  GPU 사용 가능: {torch.cuda.is_available()}")
except ImportError:
    print("  PyTorch 없음  ✗"); ok = False
sys.exit(0 if ok else 1)
PYEOF

[ -f "$TEST" ] || die "리더보드 입력 없음: $TEST"
# 문제 본문에 줄바꿈이 들어 있어 wc -l 은 과다 계수한다. csv 로 세야 한다.
say "리더보드 입력: $($PY -c "import csv,sys;print(sum(1 for _ in csv.DictReader(open(sys.argv[1],encoding='utf-8-sig'))))" "$TEST")문제"

ADAPTERS=(
  hybrid_3145:checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter
  hybrid_3244:checkpoints/hybrid_3244_r8_qv_lr2e6_e1/final_adapter
  external_3000:checkpoints/external_3000_r8_qv_lr2e6_e1/final_adapter
  hybrid_4145:checkpoints/hybrid_4145_r8_qv_lr1p5e6_e1/final_adapter
)
for a in "${ADAPTERS[@]}"; do
  p="${a#*:}"; [ -d "$ROOT/$p" ] || die "어댑터 없음: $p"
done
[ -d /workspace/models/Qwen2.5-3B-Instruct ] || die "베이스 모델 없음 (규칙상 Qwen2.5-3B-Instruct 고정)"
say "어댑터 4종 + 베이스 모델 확인"

if [ "$MODE" = "compose" ]; then
  # ───────────────────────────────────────────────────────────────────────────
  # compose — 기존 산출물에서 제출본을 조립하고 참조 CSV 와 대조한다.
  # ───────────────────────────────────────────────────────────────────────────
  say "=== compose: 기존 산출물 -> 제출본 (결정론적) ==="

  say "[1/2] 챔피언 체인 재현 검증 (5-voter -> support4 SC -> TIR 게이트)"
  $PY scripts/rebuild_chain.py --extractor v1 --verify-only \
      > "$LOGS/compose_chain.log" 2>&1 || die "rebuild_chain 실패 ($LOGS/compose_chain.log)"
  # 표 머리글에도 '불일치' 라는 낱말이 있다. 실제 실패는 '<수>개 불일치' 형태다.
  grep -qE "[0-9]+개 불일치" "$LOGS/compose_chain.log" && {
      grep -nE "[0-9]+개 불일치" "$LOGS/compose_chain.log"; die "체인 재현 불일치"; }
  sed -n '1,12p' "$LOGS/compose_chain.log"

  say "[2/4] merged16(656) 재현 검증"
  $PY scripts/build_merged16_submission.py --verify-only \
      > "$LOGS/compose_merged16.log" 2>&1
  rc=$?
  sed -n '1,12p' "$LOGS/compose_merged16.log"
  [ $rc -eq 0 ] || die "merged16 재현 불일치 ($LOGS/compose_merged16.log)"

  say "[3/4] pool24 v3mc2(660, 현행 챔피언) 재현 검증"
  $PY scripts/build_merged16_submission.py \
      --vote3-pools "outputs/tir_sc8_831_vote3_to60.jsonl,outputs/tir_repair1_831_gate282.jsonl,outputs/tir_nocode_831_gate282.jsonl" \
      --vote3-min-count 2 \
      --reference submissions/submission_pool24_v3mc2.csv --verify-only \
      > "$LOGS/compose_pool24.log" 2>&1
  rc=$?
  sed -n '1,8p' "$LOGS/compose_pool24.log"
  [ $rc -eq 0 ] || die "pool24 v3mc2(660) 재현 불일치 ($LOGS/compose_pool24.log)"

  # ck 게이트+코드가드본은 산출물이 있을 때만 검증 (제출/채택 여부와 무관하게 재현성 보존)
  if [ -f "$ROOT/submissions/submission_ck150_gate5_sup4_codeguard.csv" ]; then
    say "[4/4] ck150 게이트+코드가드 재현 검증"
    $PY scripts/build_ck_gate_submission.py --output /dev/null \
        --reference submissions/submission_ck150_gate5_sup4_codeguard.csv --verify-only \
        > "$LOGS/compose_ckgate.log" 2>&1
    rc=$?
    sed -n '1,6p' "$LOGS/compose_ckgate.log"
    [ $rc -eq 0 ] || die "ck 게이트본 재현 불일치 ($LOGS/compose_ckgate.log)"
  fi

  if [ -f "$ROOT/submissions/submission_final_gate425.csv" ]; then
    say "[5/5] 게이트 v3 (N=64, 0.425) 재현 검증"
    $PY scripts/build_final_union_submission.py --output /dev/null \
        --reference submissions/submission_final_gate425.csv --verify-only \
        > "$LOGS/compose_gate425.log" 2>&1
    rc=$?
    tail -2 "$LOGS/compose_gate425.log"
    [ $rc -eq 0 ] || die "게이트 v3 재현 불일치 ($LOGS/compose_gate425.log)"
  fi

  say "=== compose 통과: 제출본이 규칙대로 조립됨을 확인 ==="
  exit 0
fi

[ "$MODE" = "full" ] || die "모드는 compose 또는 full"

# ─────────────────────────────────────────────────────────────────────────────
# full — 빈 상태에서 새로 생성한다.
#
# GPU 점유 순서가 중요하다. 32GB pod 에서 vLLM 서버(0.90)와 HF 추론을 동시에 띄우면
# OOM 이 난다 (CONTEXT 22절). voter/SC 는 HF, TIR 은 서버가 필요하므로 직렬화한다.
# ─────────────────────────────────────────────────────────────────────────────
LIMIT_ARG=""
[ -n "$SMOKE" ] && { LIMIT_ARG="--limit $SMOKE"; say "*** 스모크 모드: $SMOKE 문제만 ***"; }

# 재생성 경로에서는 챔피언·표수·risky 도 **새 산출물**에서 계산해야 한다.
# 이 인자를 빼면 옛 outputs/*.jsonl 을 읽어 TIR 풀만 새 것이 섞인다 (조용한 불일치).
REPRO_ARGS=(
  --voters "$REPRO/hybrid_3145_leaderboard_retry2048.jsonl"
           "$REPRO/hybrid_3244_leaderboard_retry2048.jsonl"
           "$REPRO/external_3000_leaderboard_retry2048.jsonl"
           "$REPRO/hybrid_4145_leaderboard_retry2048.jsonl"
           "$REPRO/hybrid_3145_verify_leaderboard_retry2048.jsonl"
  --sc-files "$REPRO/self_consistency_hybrid3145_n8_leaderboard_support1to3.jsonl"
             "$REPRO/self_consistency_hybrid3145_n8_leaderboard_support4.jsonl"
             "$REPRO/self_consistency_hybrid3145_n8_leaderboard_support5.jsonl"
)

server_up() { curl -s -m 3 -o /dev/null http://localhost:8000/v1/models; }
stop_server() {
  local pids; pids=$(pgrep -f "vllm.entrypoints.openai.api_server" || true)
  [ -n "$pids" ] || return 0
  say "vLLM 서버 종료 (PID: $pids)"
  # shellcheck disable=SC2086
  kill $pids; sleep 15
}
start_server() {
  server_up && { say "vLLM 서버 이미 떠 있음"; return 0; }
  say "vLLM 서버 기동 (12~13분 소요)"
  VLLM_GPU_FRAC=0.90 setsid nohup bash "$ROOT/scripts/vllm_server.sh" \
      > "$ROOT/$LOGS/vllm.log" 2>&1 < /dev/null &
  for _ in $(seq 1 100); do sleep 15; server_up && { say "서버 준비 완료"; return 0; }; done
  die "서버 기동 타임아웃 ($LOGS/vllm.log)"
}

# ── Stage 1: 5-voter 예측 (831문제) ────────────────────────────────────────────
# voter5(hybrid_3145_verify)는 voter1 과 **같은 가중치**이고 프롬프트만 다르다.
# 데이터 증강보다 추론 시점 프롬프트가 더 큰 레버리지였다는 CONTEXT 결론 4번의 산물이다.
if ! skip_stage voters; then
  say "=== Stage 1: 5-voter 생성 (HF, GPU 단독) ==="
  stop_server
  gen_voter() {
    local name="$1" adapter="$2" style="$3"
    local out="$REPRO/${name}_leaderboard_retry2048.jsonl"
    [ -f "$out" ] && { say "SKIP voter $name"; return 0; }
    say "voter $name (style=$style) 생성 중"
    $PY scripts/baseline.py \
      --adapter-path "$adapter" --input "$TEST" --output "$out" \
      --prompt-style "$style" \
      --max-new-tokens 1024 --retry-max-new-tokens 2048 \
      --batch-size 64 --retry-batch-size 16 $LIMIT_ARG \
      > "$LOGS/voter_${name}.log" 2>&1 || { tail -20 "$LOGS/voter_${name}.log"; die "voter $name 실패"; }
    say "voter $name 완료 ($(wc -l < "$out")행)"
  }
  for a in "${ADAPTERS[@]}"; do gen_voter "${a%%:*}" "${a#*:}" default; done
  gen_voter hybrid_3145_verify checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter verify
fi

# ── Stage 2: SC N=8 (support tier 별) ─────────────────────────────────────────
# support tier 는 5-voter 합의도라 Stage 1 이 끝나야 확정된다.
if ! skip_stage sc; then
  say "=== Stage 2: self-consistency N=8 생성 (HF, GPU 단독) ==="
  stop_server
  # ⚠️ extend_self_consistency_samples 는 소스의 `baseline_support` 필드로 tier 를 거른다.
  # voter 하나의 산출물에는 그 필드가 없어서(support 는 5-voter 합의도다) 그대로 넘기면
  # 전 행이 조용히 탈락해 **빈 파일**이 나온다. 5-voter 를 먼저 합쳐야 한다.
  SRC="$REPRO/sc_source.jsonl"
  if [ ! -f "$SRC" ]; then
    say "5-voter 합의 -> SC 소스 생성"
    $PY scripts/build_sc_source.py \
      --voters "$REPRO/hybrid_3145_leaderboard_retry2048.jsonl" \
               "$REPRO/hybrid_3244_leaderboard_retry2048.jsonl" \
               "$REPRO/external_3000_leaderboard_retry2048.jsonl" \
               "$REPRO/hybrid_4145_leaderboard_retry2048.jsonl" \
               "$REPRO/hybrid_3145_verify_leaderboard_retry2048.jsonl" \
      --output "$SRC" > "$LOGS/sc_source.log" 2>&1 || { cat "$LOGS/sc_source.log"; die "SC 소스 생성 실패"; }
    cat "$LOGS/sc_source.log"
  fi
  [ -s "$SRC" ] || die "SC 소스가 비었다: $SRC"
  gen_sc() {
    local tag="$1" lo="$2" hi="$3"
    local out="$REPRO/self_consistency_hybrid3145_n8_leaderboard_${tag}.jsonl"
    [ -f "$out" ] && { say "SKIP sc $tag"; return 0; }
    say "SC $tag (support $lo~$hi) 생성 중"
    $PY scripts/extend_self_consistency_samples.py \
      --source "$SRC" \
      --adapter-path checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter \
      --output "$out" --num-samples 8 \
      --min-baseline-support "$lo" --max-baseline-support "$hi" \
      --max-new-tokens 1024 --batch-size 16 \
      > "$LOGS/sc_${tag}.log" 2>&1 || { tail -20 "$LOGS/sc_${tag}.log"; die "SC $tag 실패"; }
    say "SC $tag 완료 ($(wc -l < "$out")행)"
  }
  gen_sc support1to3 1 3
  gen_sc support4    4 4
  gen_sc support5    5 5
fi

# ── Stage 3: TIR 풀 2종 (게이트 282문제) ──────────────────────────────────────
# 게이트는 SC 표수에서 유도되므로 Stage 2 이후에만 확정된다.
# 풀을 두 개 만들어 **합치는** 것이 핵심이다 (교체가 아니다, CONTEXT B절).
if ! skip_stage tir; then
  say "=== Stage 3: TIR 풀 생성 (vLLM 서버 필요) ==="
  $PY scripts/build_merged16_submission.py --export-gates "$REPRO" \
      "${REPRO_ARGS[@]}" \
      > "$LOGS/gates.log" 2>&1 || { cat "$LOGS/gates.log"; die "게이트 산출 실패"; }
  cat "$LOGS/gates.log"
  GATE="$REPRO/repro_831_gate282.csv"
  start_server
  gen_tir() {
    local tag="$1" seed="$2"
    local out="$REPRO/tir_${tag}_gate282.jsonl"
    [ -f "$out" ] && { say "SKIP tir $tag"; return 0; }
    say "TIR 풀 $tag (seed=$seed) 생성 중 — 코드 리페어 + 미생성 재시도"
    $VPY scripts/tir_repair_client.py \
      --input "$GATE" --output "$out" --model hybrid3145 \
      --num-samples 8 --repair-rounds 1 --nocode-retries 1 \
      --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed "$seed" \
      > "$LOGS/tir_${tag}.log" 2>&1 || { tail -20 "$LOGS/tir_${tag}.log"; die "TIR $tag 실패"; }
    tail -2 "$LOGS/tir_${tag}.log"
  }
  gen_tir poolA 20260828
  gen_tir poolB 20260829
fi

# ── Stage 4: 병합 -> 최종 제출본 ──────────────────────────────────────────────
say "=== Stage 4: 병합 -> 최종 제출본 ==="
OUT_CSV="submissions/submission_reproduced.csv"
$PY scripts/build_merged16_submission.py \
  --vote3-pools  "$REPRO/tir_poolA_gate282.jsonl,$REPRO/tir_poolB_gate282.jsonl" \
  --vote45-pools "$REPRO/tir_poolA_gate282.jsonl,$REPRO/tir_poolB_gate282.jsonl" \
  "${REPRO_ARGS[@]}" \
  --no-reference --out "$OUT_CSV" \
  > "$LOGS/merge.log" 2>&1
sed -n '1,12p' "$LOGS/merge.log"
[ -f "$ROOT/$OUT_CSV" ] || die "제출본 생성 실패 ($LOGS/merge.log)"

# 무결성 — 행 수, ID 중복, 정수 여부. 제출 전 항상 이걸 통과해야 한다.
$PY - "$ROOT/$OUT_CSV" <<'PYEOF'
import csv, sys
rows = list(csv.DictReader(open(sys.argv[1], encoding="utf-8-sig")))
ids = [r["id"] for r in rows]
assert len(rows) == 831, f"행 수 {len(rows)} != 831"
assert len(set(ids)) == 831, "ID 중복"
for r in rows:
    int(r["answer"])
print(f"무결성 통과: {len(rows)}행, ID 중복 0, 전부 정수")
PYEOF

say "=== 완료: $OUT_CSV ==="
say "주의: full 모드 산출물은 샘플링이 확률적이라 656 과 답이 일부 다르다 (정상)."
say "      규칙 재현은 'compose' 모드로 증명한다."
