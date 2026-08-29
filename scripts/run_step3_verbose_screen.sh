#!/usr/bin/env bash
# Step 3-1 — verbose distill 기준으로 GRPO 학습 풀을 다시 스크리닝한다.
#
# 왜 다시 스크리닝하나: 기존 grpo_passrate_94는 hybrid_3145의 pass rate로 고른 것이다.
# GRPO는 "그 정책이 부분적으로만 성공하는(25~75%) 문제"에서만 그레이디언트가 생기므로,
# 출발점 정책이 verbose distill로 바뀌면 타겟 문제 집합도 바뀌어야 한다.
#
# pool600은 누수 검증 완료: fixed200/holdout500 겹침 0, hybrid3145 SFT 데이터 겹침 0.
set -u
cd /workspace/DLC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

OUT=outputs/step3_verbose_n8_pool600_seed20260903.jsonl
if [ -f "${OUT}" ]; then echo "[$(date +%H:%M:%S)] SCREEN-SKIP (이미 있음)"; exit 0; fi

echo "[$(date +%H:%M:%S)] SCREEN-START verbose distill on pool600 (600 x 8 = 4800 generations)"
python3 scripts/evaluate_self_consistency.py \
  --questions data/selector/selector_pool_600.csv \
  --predictions outputs/selector_pool600_hybrid3145.csv \
  --adapter-path checkpoints/verbose_distill_r8_qv_lr2e6_e1/final_adapter \
  --output "${OUT}" \
  --subset-size 600 \
  --num-samples 8 \
  --min-count 4 \
  --batch-size 16 \
  --max-new-tokens 768 \
  --temperature 0.7 --top-p 0.95 \
  --model-seed 20260903 > logs/step3_verbose_screen.log 2>&1
rc=$?
if [ $rc -ne 0 ]; then
  echo "[$(date +%H:%M:%S)] SCREEN-FAILED rc=${rc}"; tail -20 logs/step3_verbose_screen.log; exit 1
fi
echo "[$(date +%H:%M:%S)] SCREEN-DONE -> ${OUT}"
