#!/usr/bin/env bash
# Step 2 — GRPO96 · verbose distill의 답이 갈린 51문제에만 추가 stochastic N=8을 양쪽 모델로 생성한다.
# 기존 풀(각 계보 N=8, seed 20260826/20260827)과 합치면 이 51문제는 계보당 16샘플, 총 32샘플이 된다.
# 생성 파라미터는 기존 풀과 동일하게 맞춘다: temp 0.7 / top_p 0.95 / max_new_tokens 768.
# 순차 실행한다 — 3개 병렬로 돌렸다가 배치당 471초로 오히려 느려진 전례가 있다.
set -u
cd /workspace/DLC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

Q=data/holdout/holdout464_gv_disagree51.csv
P=outputs/gv_disagree51_grpo96_det_predictions.csv

run() {
  local name="$1" adapter="$2" seed="$3"
  local out="outputs/step2_${name}_n8_disagree51_seed${seed}.jsonl"
  if [ -f "${out}" ]; then echo "[$(date +%H:%M:%S)] STEP2-SKIP ${name}"; return 0; fi
  echo "[$(date +%H:%M:%S)] STEP2-START ${name} seed=${seed}"
  python3 scripts/evaluate_self_consistency.py \
    --questions "${Q}" \
    --predictions "${P}" \
    --adapter-path "${adapter}" \
    --output "${out}" \
    --subset-size 51 \
    --num-samples 8 \
    --min-count 4 \
    --batch-size 16 \
    --max-new-tokens 768 \
    --temperature 0.7 --top-p 0.95 \
    --model-seed "${seed}" > "logs/step2_${name}.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] STEP2-FAILED ${name} rc=${rc}"; tail -20 "logs/step2_${name}.log"; return 1
  fi
  echo "[$(date +%H:%M:%S)] STEP2-DONE ${name} -> ${out}"
}

run grpo96  checkpoints/grpo_3145_passrate94_r8_qv_lr1e6_steps96_g8/final_adapter 20260901
run verbose checkpoints/verbose_distill_r8_qv_lr2e6_e1/final_adapter            20260902
echo "[$(date +%H:%M:%S)] STEP2-ALL-DONE"
