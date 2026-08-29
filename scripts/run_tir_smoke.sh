#!/usr/bin/env bash
# TIR 스모크 — 학습 없이 프롬프트 + 로컬 코드 실행만으로 신호가 있는지 본다.
#
# ⚠️ 추론 시점 코드 실행은 규정 회색지대다. 운영진 확인 전까지 이 산출물로 제출하지 않는다.
#
# 1) unsolved68 — N=8을 8번 전부 틀린 완전 미해결 문제. 여기서 열리면 진짜 신규 능력이다.
# 2) gate120   — holdout464 앞 120문제. 기존에 풀던 걸 망가뜨리는지(회귀) 확인.
#    비교 기준: hybrid_3145 기준선 95/120 (retry 없는 조건에서는 더 낮음, 아래 주석 참고)
set -u
cd /workspace/DLC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

ADAPTER=checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter

echo "[$(date +%H:%M:%S)] TIR-START unsolved68"
python3 scripts/tir_inference.py \
  --input data/holdout/official_holdout_464_unsolved68.jsonl \
  --output outputs/tir_unsolved68_hybrid3145.jsonl \
  --adapter-path "${ADAPTER}" \
  --batch-size 16 --max-new-tokens 768 --final-max-new-tokens 512 \
  > logs/tir_unsolved68.log 2>&1
echo "[$(date +%H:%M:%S)] TIR-DONE unsolved68 rc=$?"
tail -2 logs/tir_unsolved68.log

echo "[$(date +%H:%M:%S)] TIR-START gate120"
python3 scripts/tir_inference.py \
  --input data/holdout/holdout464_gate120.csv \
  --output outputs/tir_gate120_hybrid3145.jsonl \
  --adapter-path "${ADAPTER}" \
  --batch-size 16 --max-new-tokens 768 --final-max-new-tokens 512 \
  > logs/tir_gate120.log 2>&1
echo "[$(date +%H:%M:%S)] TIR-DONE gate120 rc=$?"
tail -2 logs/tir_gate120.log

echo "[$(date +%H:%M:%S)] TIR-ALL-DONE"
