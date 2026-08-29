#!/usr/bin/env bash
# 1.5순위: 학습 강도를 올려 verbose distill 데이터가 흡수되는지 확인한다.
#
# 판정 구조 (2단계):
#   1차 게이트 (빠름, ~5분) — holdout464 앞 120문제에서 `**Approach:**` 마커 출현율과 응답 길이.
#      근거: rank64 스윕에서 학습 강도↑ → 응답 길이↓가 단조였고 점수도 같이 떨어졌다.
#      즉 "길이/스타일이 데이터 쪽으로 움직이는가"가 흡수의 직접 지표다.
#   2차 (전체, ~30분) — 1차에서 흡수 신호가 있는 설정만 holdout464 464문제 전량 평가.
#
# 로그는 logs/<name>.train.log / <name>.gate.log 에 남긴다 (버퍼링 없이 관측 가능).
set -u
cd /workspace/DLC
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p logs

DATA=data/processed/hybrid_verbose_distill.jsonl
HOLDOUT=data/holdout/official_holdout_464_clean.csv
GATE_N=120

train_one() {
  local name="$1" rank="$2" targets="$3" lr="$4" epochs="$5"
  local ckpt="checkpoints/${name}"
  echo "[$(date +%H:%M:%S)] TRAIN-START ${name} rank=${rank} targets=${targets} lr=${lr} ep=${epochs}"
  if [ -f "${ckpt}/final_adapter/adapter_model.safetensors" ]; then
    echo "[$(date +%H:%M:%S)] TRAIN-SKIP ${name} (already exists)"
    return 0
  fi
  python3 scripts/train_qlora.py \
    --data "${DATA}" \
    --output-dir "${ckpt}" \
    --learning-rate "${lr}" \
    --lora-rank "${rank}" \
    --target-modules "${targets}" \
    --epochs "${epochs}" \
    --batch-size 8 --gradient-accumulation 2 \
    --max-minutes 90 --seed 2026 > "logs/${name}.train.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] TRAIN-FAILED ${name} rc=${rc}"
    tail -20 "logs/${name}.train.log"
    return 1
  fi
  echo "[$(date +%H:%M:%S)] TRAIN-DONE ${name} $(grep -o "'elapsed_seconds': [0-9.]*" ${ckpt}/training_metadata.json 2>/dev/null || echo '')"
}

gate_one() {
  local name="$1"
  local out="outputs/${name}_gate${GATE_N}.jsonl"
  echo "[$(date +%H:%M:%S)] GATE-START ${name}"
  python3 scripts/baseline.py \
    --adapter-path "checkpoints/${name}/final_adapter" \
    --input "${HOLDOUT}" --limit "${GATE_N}" \
    --output "${out}" \
    --max-new-tokens 1024 --batch-size 60 > "logs/${name}.gate.log" 2>&1
  local rc=$?
  if [ $rc -ne 0 ]; then
    echo "[$(date +%H:%M:%S)] GATE-FAILED ${name} rc=${rc}"; tail -20 "logs/${name}.gate.log"; return 1
  fi
  python3 scripts/report_absorption.py "${out}" "${name}"
}

run_one() { train_one "$@" && gate_one "$1"; }

# 참조점: 기존 약한 설정(이미 학습됨)을 같은 게이트로 측정해 비교 기준을 만든다.
gate_one verbose_distill_r8_qv_lr2e6_e1

run_one verbose_r8_qv_lr2e6_e3      8  qv  2e-6 3
run_one verbose_r16_qv_lr1e5_e2     16 qv  1e-5 2
run_one verbose_r32_all_lr1e5_e3    32 all 1e-5 3

echo "[$(date +%H:%M:%S)] SWEEP-DONE"
