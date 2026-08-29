#!/usr/bin/env bash
# run_voter_holdout464.sh의 병렬판.
# A100 80GB에서 3B 모델 3개를 동시에 올려도 메모리는 넉넉하고(각 ~10GB),
# 디코딩은 weight bandwidth 바운드라 동시 실행이 순차 실행보다 총 처리량이 높다.
# 배치 구성(batch 64 / retry 16)은 이미 완료된 hybrid_3244 실행과 동일하게 유지한다.
set -u
cd /workspace/DLC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

HOLDOUT=data/holdout/official_holdout_464_clean.csv

gen() {
  local name="$1" adapter="$2" style="$3"
  local out="outputs/${name}_holdout464_retry2048.jsonl"
  if [ -f "${out}" ]; then echo "[$(date +%H:%M:%S)] VOTER-SKIP ${name}"; return 0; fi
  echo "[$(date +%H:%M:%S)] VOTER-START ${name} (style=${style})"
  python3 scripts/baseline.py \
    --adapter-path "${adapter}" \
    --input "${HOLDOUT}" \
    --output "${out}" \
    --prompt-style "${style}" \
    --max-new-tokens 1024 --retry-max-new-tokens 2048 \
    --retry-batch-size 16 --batch-size 64 > "logs/${name}_holdout464.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] VOTER-FAILED ${name} rc=${rc}"; tail -20 "logs/${name}_holdout464.log"; return 1
  fi
  python3 -c "
import json
rows=[json.loads(l) for l in open('${out}')]
print('[$(date +%H:%M:%S)] VOTER-DONE ${name}:', sum(1 for r in rows if r.get('correct')), '/', len(rows))
"
}

gen external_3000 checkpoints/external_3000_r8_qv_lr2e6_e1/final_adapter default &
P1=$!
gen hybrid_4145 checkpoints/hybrid_4145_r8_qv_lr1p5e6_e1/final_adapter default &
P2=$!
gen hybrid_3145_verify checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter verify &
P3=$!

wait $P1 $P2 $P3
echo "[$(date +%H:%M:%S)] VOTERS-DONE"
