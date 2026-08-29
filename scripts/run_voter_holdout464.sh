#!/usr/bin/env bash
# 현행 챔피언(5-voter + SC support4, Public 0.74969)을 holdout464에서 직접 채점하기 위해
# 빠져 있는 4개 voter의 holdout464 예측을 생성한다.
#
# 왜 필요한가: 새 3계보 규칙은 holdout464에서 357→371(+14)이다. 831로 환산하면 약 +25인데,
# 챔피언은 Public에서 hybrid3145(594/831) 대비 +29다. 즉 새 규칙이 챔피언을 이긴다는 보장이 없다.
# 두 지표가 서로 다른 셋에서 나온 것이라 직접 비교가 불가능한 것이 문제이므로,
# holdout464를 공통 기준으로 만든다. (CONTEXT "세션 전략 재점검" 5번의 방침)
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
import json,sys
rows=[json.loads(l) for l in open('${out}')]
print('[$(date +%H:%M:%S)] VOTER-DONE ${name}:', sum(1 for r in rows if r.get('correct')), '/', len(rows))
"
}

gen hybrid_3244   checkpoints/hybrid_3244_r8_qv_lr2e6_e1/final_adapter    default
gen external_3000 checkpoints/external_3000_r8_qv_lr2e6_e1/final_adapter  default
gen hybrid_4145   checkpoints/hybrid_4145_r8_qv_lr1p5e6_e1/final_adapter  default
gen hybrid_3145_verify checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter verify

echo "[$(date +%H:%M:%S)] VOTERS-DONE"
