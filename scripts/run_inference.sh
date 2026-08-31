#!/usr/bin/env bash
# 최종 테스트 2,000 — 제출 조립에 쓰는 추론 재료 전체 스택 생성.
# 입력/출력 기본값: data/deep_chal_math_dataset_test.csv -> outputs/final
# (다른 입력으로 돌리려면 MEGA_IN/MEGA_OUT 환경변수로 재지정)
# 단계: 보이터5 -> SC N=8(응답 포함) -> 밴드 산출 -> TIR 풀 -> ck150 N=8/N=64 -> (조립은 별도)
# 각 단계는 산출물이 완결(행수 = 입력 문항 수)이면 건너뛴다 (재개 안전).
set -euo pipefail
R=$(cd "$(dirname "$0")/.." && pwd)
V=/workspace/venv-vllm/bin/python
SYS_PY=${SYS_PY:-python3}   # mega_bands 용 시스템 파이썬 (표준 라이브러리만 쓴다)
IN=${MEGA_IN:-data/deep_chal_math_dataset_test.csv}
O=${MEGA_OUT:-outputs/final}
LOGS=logs/inference
cd "$R"
mkdir -p "$O" "$LOGS"
say(){ echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] $*"; }
die(){ echo "[실패] $*" >&2; exit 1; }

# CSV 행수는 wc -l 로 세면 안 된다 — 문제 본문에 줄바꿈이 있어 과다 계수된다.
csv_rows(){ $SYS_PY -c "import csv,sys;print(sum(1 for _ in csv.DictReader(open(sys.argv[1],encoding='utf-8-sig'))))" "$1"; }
jsonl_rows(){ if [ -f "$1" ]; then wc -l < "$1" | tr -d ' '; else echo 0; fi; }
# 완결 판정: 부분 생성된 파일(중단된 실행)은 완료로 치지 않고 다시 만든다.
is_done(){ [ "$(jsonl_rows "$1")" -ge "$2" ]; }

# 실패를 침묵시키지 않는다: 전체 출력을 로그로 받고, 실패 시 꼬리를 보여주고 멈춘다.
run_logged(){ # logname cmd...
  local name="$1" log="$LOGS/$1"; shift
  "$@" > "$log" 2>&1 || { tail -20 "$log" >&2; die "$name (전체 로그: $log)"; }
  tail -1 "$log"
}

NPROB=$(csv_rows "$IN")
say "입력: $IN (${NPROB}문제) -> $O"

say "=== 1) 보이터 5종 greedy ==="
run_voter(){ # name model style
  is_done "$O/voter_$1.jsonl" "$NPROB" && { say "voter_$1 완결 — 건너뜀"; return 0; }
  run_logged "voter_$1.log" $V scripts/gen_client.py --input $IN --output $O/voter_$1.jsonl \
    --model $2 --prompt-style $3 --request-workers 96
}
run_voter hybrid3145 hybrid3145 default
run_voter h3244      h3244      default
run_voter ext3000    ext3000    default
run_voter h4145      h4145      default
run_voter verify     hybrid3145 verify

say "=== 2) hybrid SC N=8 (응답 포함) ==="
if is_done "$O/sc_hybrid_n8.jsonl" "$NPROB"; then
  say "sc_hybrid_n8 완결 — 건너뜀"
else
  run_logged sc_hybrid_n8.log $V scripts/screen_grpo_passrate.py --input $IN --output $O/sc_hybrid_n8.jsonl \
    --model hybrid3145 --num-samples 8 --temperature 0.7 --top-p 0.95 \
    --max-new-tokens 1024 --request-workers 64 --seed 20260828 --resume --save-responses
fi

say "=== 3) 밴드 산출 (표수/서포트/risky) ==="
MEGA_IN=$IN MEGA_OUT=$O $SYS_PY scripts/mega_bands.py

say "=== 4) TIR 풀 (표수<=3 밴드 3종 + vote45-risky 2종) ==="
tir(){ # out input seed nocode
  is_done "$O/$1" "$(csv_rows "$2")" && { say "$1 완결 — 건너뜀"; return 0; }
  run_logged "tir_$1.log" $V scripts/tir_repair_client.py --input $2 --output $O/$1 \
    --model "${TIR_MODEL:-hybrid3145}" --num-samples 8 --repair-rounds 1 --nocode-retries $4 \
    --exec-timeout 60 --exec-workers "${TIR_EXEC_WORKERS:-64}" \
    --request-workers "${TIR_REQ_WORKERS:-64}" --seed $3
}
tir tir_a100_vote3.jsonl $O/mega_vote3.csv 20260931 0
tir tir_r1_vote3.jsonl   $O/mega_vote3.csv 20260932 0
tir tir_nc_vote3.jsonl   $O/mega_vote3.csv 20260933 1
tir tir_a100_v45r.jsonl  $O/mega_vote45_risky.csv 20260934 0
tir tir_nc_v45r.jsonl    $O/mega_vote45_risky.csv 20260935 1

say "=== 5) ck150 N=8 / N=64 (support<=4) ==="
NSUP4=$(csv_rows "$O/mega_sup_le4.csv")
if is_done "$O/ck150_n8_sup4.jsonl" "$NSUP4"; then
  say "ck150_n8_sup4 완결 — 건너뜀"
else
  run_logged ck150_n8_sup4.log $V scripts/screen_grpo_passrate.py --input $O/mega_sup_le4.csv --output $O/ck150_n8_sup4.jsonl \
    --model ck150 --num-samples 8 --temperature 0.7 --top-p 0.95 \
    --max-new-tokens 1024 --request-workers 64 --seed 20260920 --resume
fi
if [ "${SKIP_N64:-0}" = "1" ]; then
  say "N=64 건너뜀 (SKIP_N64=1 — c672/c623 조립에 불필요)"
elif is_done "$O/ck150_n64_sup4.jsonl" "$NSUP4"; then
  say "ck150_n64_sup4 완결 — 건너뜀"
else
  run_logged ck150_n64_sup4.log $V scripts/screen_grpo_passrate.py --input $O/mega_sup_le4.csv --output $O/ck150_n64_sup4.jsonl \
    --model ck150 --num-samples 64 --temperature 0.7 --top-p 0.95 \
    --max-new-tokens 1024 --request-workers 48 --seed 20260922 --resume
fi

say "=== 6) 672 층 재료 (fs 포인터 게이트: fs3 greedy + fs3 N=8 + pool16) ==="
if is_done "$O/fs3_greedy.jsonl" "$NPROB"; then say "fs3_greedy 완결 — 건너뜀"; else
  run_logged fs3_greedy.log $V scripts/gen_fewshot_client.py --input $IN --output $O/fs3_greedy.jsonl \
    --exemplars outputs/fewshot_exemplars3.json --model hybrid3145 \
    --request-workers 64
fi
if is_done "$O/fs3_n8.jsonl" "$NPROB"; then say "fs3_n8 완결 — 건너뜀"; else
  run_logged fs3_n8.log $V scripts/gen_fewshot_client.py --input $IN --output $O/fs3_n8.jsonl \
    --exemplars outputs/fewshot_exemplars3.json --model hybrid3145 \
    --num-samples 8 --seed 20260925 --request-workers 64
fi
if is_done "$O/h3145_n8lp.jsonl" "$NPROB"; then say "h3145_n8lp 완결 — 건너뜀"; else
  run_logged h3145_n8lp.log $V scripts/gen_n8_logprobs.py --input $IN --output $O/h3145_n8lp.jsonl \
    --model hybrid3145 --seed 20260924 --request-workers 48
fi
if is_done "$O/ck150_n8lp.jsonl" "$NPROB"; then say "ck150_n8lp 완결 — 건너뜀"; else
  run_logged ck150_n8lp.log $V scripts/gen_n8_logprobs.py --input $IN --output $O/ck150_n8lp.jsonl \
    --model ck150 --seed 20260924 --request-workers 48
fi
say "=== 생성 전체 완료 — 조립: scripts/compose_final_submissions.py ==="
