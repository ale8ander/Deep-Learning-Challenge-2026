#!/usr/bin/env bash
# 메가 홀드아웃 2,000 — 최종 제출 선정용 전체 스택 생성.
# 단계: 보이터5 -> SC N=8(응답 포함) -> 밴드 산출 -> TIR 풀 -> ck150 N=8/N=64 -> (채점은 별도)
# 각 단계는 산출물이 있으면 건너뛴다 (재개 안전).
set -u
R=/workspace/DLC
V=/workspace/venv-vllm/bin/python
IN=data/holdout/mega_holdout_2000.csv
O=outputs/mega
mkdir -p $R/$O
cd $R
say(){ echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] $*"; }

say "=== 1) 보이터 5종 greedy ==="
run_voter(){ # name model style
  [ -s "$O/voter_$1.jsonl" ] && { say "voter_$1 있음 — 건너뜀"; return; }
  $V scripts/gen_client.py --input $IN --output $O/voter_$1.jsonl \
    --model $2 --prompt-style $3 --request-workers 96 2>&1 | tail -1
}
run_voter hybrid3145 hybrid3145 default
run_voter h3244      h3244      default
run_voter ext3000    ext3000    default
run_voter h4145      h4145      default
run_voter verify     hybrid3145 verify

say "=== 2) hybrid SC N=8 (응답 포함) ==="
if [ ! -s "$O/sc_hybrid_n8.jsonl" ] || [ "$(wc -l < $O/sc_hybrid_n8.jsonl)" -lt 2000 ]; then
  $V scripts/screen_grpo_passrate.py --input $IN --output $O/sc_hybrid_n8.jsonl \
    --model hybrid3145 --num-samples 8 --temperature 0.7 --top-p 0.95 \
    --max-new-tokens 1024 --request-workers 64 --seed 20260828 --resume --save-responses 2>&1 | tail -1
fi

say "=== 3) 밴드 산출 (표수/서포트/risky) ==="
/usr/bin/python3 scripts/mega_bands.py

say "=== 4) TIR 풀 (표수<=3 밴드 3종 + vote45-risky 2종) ==="
tir(){ # out input seed nocode
  [ -s "$O/$1" ] && { say "$1 있음 — 건너뜀"; return; }
  $V scripts/tir_repair_client.py --input $2 --output $O/$1 \
    --model tirsft --num-samples 8 --repair-rounds 1 --nocode-retries $4 \
    --exec-timeout 60 --exec-workers 64 --request-workers 64 --seed $3 2>&1 | tail -1
}
tir tir_a100_vote3.jsonl $O/mega_vote3.csv 20260931 0
tir tir_r1_vote3.jsonl   $O/mega_vote3.csv 20260932 0
tir tir_nc_vote3.jsonl   $O/mega_vote3.csv 20260933 1
tir tir_a100_v45r.jsonl  $O/mega_vote45_risky.csv 20260934 0
tir tir_nc_v45r.jsonl    $O/mega_vote45_risky.csv 20260935 1

say "=== 5) ck150 N=8 / N=64 (support<=4) ==="
if [ ! -s "$O/ck150_n8_sup4.jsonl" ]; then
  $V scripts/screen_grpo_passrate.py --input $O/mega_sup_le4.csv --output $O/ck150_n8_sup4.jsonl \
    --model ck150 --num-samples 8 --temperature 0.7 --top-p 0.95 \
    --max-new-tokens 1024 --request-workers 64 --seed 20260920 --resume 2>&1 | tail -1
fi
if [ ! -s "$O/ck150_n64_sup4.jsonl" ]; then
  $V scripts/screen_grpo_passrate.py --input $O/mega_sup_le4.csv --output $O/ck150_n64_sup4.jsonl \
    --model ck150 --num-samples 64 --temperature 0.7 --top-p 0.95 \
    --max-new-tokens 1024 --request-workers 48 --seed 20260922 --resume 2>&1 | tail -1
fi
say "=== 생성 전체 완료 — 채점: scripts/mega_score.py ==="
