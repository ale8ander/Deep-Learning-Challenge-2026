#!/usr/bin/env bash
# A100 80GB 야간 연쇄. pod 스왑 + bootstrap_pod.sh 이후 실행.
#
# 80GB 의 핵심 이점: 서버를 **0.45(36GB)** 로 띄우면 학습·teacher 생성과 **공존**한다.
# 5090(32GB)에서는 매번 서버를 내렸다 올리느라 13분 x N 을 냈다. 그 세금이 사라진다.
#
# 순서 (제출에 가까운 것부터 — C-4 준수: 홀드아웃 판정이 항상 831 생성보다 먼저):
#   1) R1 증류 홀드아웃 판정      (어댑터는 5090 에서 학습 완료, 볼륨에 보존)
#   2) P1 표수4~5 밴드 풀 + 스윕  ← **마지막 미개척 제출 레버**
#   3) 32B teacher 생성           (밤새, 아침에 학습 여부 결정)
#
# ⚠️ 프로세스 종료는 예외 없이 PID 로 (pkill -f 는 자기 명령줄을 매칭해 자신을 죽인다)
set -u
R=/workspace/DLC
V=/workspace/venv-vllm/bin/python
say(){ echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] $*"; }
up(){ curl -s -m 3 -o /dev/null http://localhost:8000/v1/models; }

start_server() {
  up && { say "서버 이미 떠 있음"; return 0; }
  say "vLLM 서버 기동 (frac=0.45 — 학습과 공존 가능)"
  VLLM_GPU_FRAC=0.45 setsid nohup bash $R/scripts/vllm_server.sh \
    > $R/logs/vllm.log 2>&1 < /dev/null &
  for _ in $(seq 1 120); do sleep 15; up && { say "서버 준비"; return 0; }; done
  say "서버 기동 타임아웃"; return 1
}

WORKERS=$(( $(nproc) * 3 ))   # 코드 실행은 CPU 바운드가 아니라 대기 바운드다. 코어 x3.
say "exec-workers=$WORKERS (코어 $(nproc))"

start_server || exit 1

# ── 1) R1 증류 판정 ────────────────────────────────────────────────
CK=$R/checkpoints/r1_distill_r8qv_lr1e5/final_adapter
if [ -f "$CK/adapter_config.json" ]; then
  say "=== 1) R1 증류 홀드아웃 판정 ==="
  curl -s -m 60 -X POST http://localhost:8000/v1/load_lora_adapter \
    -H 'Content-Type: application/json' \
    -d "{\"lora_name\":\"r1distill\",\"lora_path\":\"$CK\"}" \
    -w " (load_lora http=%{http_code})\n"
  for M in r1distill hybrid3145; do
    o=$R/outputs/r1distill_gen_${M}_holdout87.jsonl
    [ -f "$o" ] && { say "SKIP $M"; continue; }
    $V $R/scripts/gen_client.py \
      --input $R/data/holdout/holdout464_vote3.csv --output "$o" \
      --model "$M" --max-new-tokens 2048 --retry-max-new-tokens 3072 \
      --request-workers 48 --seed 20260911 \
      > $R/logs/r1distill_gen_${M}.log 2>&1 || say "평가 $M 실패"
  done
  python3 $R/scripts/report_r1_distill.py 2>&1 | tee $R/logs/r1_distill_gate.log
else
  say "=== 1) SKIP — R1 어댑터 없음 ==="
fi

# ── 2) P1: 표수4~5 밴드 (마지막 미개척 제출 레버) ──────────────────
# 현재 이 구간 109문제(831) 중 risky>=1 인 26개만 TIR 을 쓴다. 나머지는 손대지 않았다.
# 831 환산 오답이 이 구간에 34개 있다.
say "=== 2) 표수4~5 밴드 풀 생성 (홀드아웃 69문제) ==="
BAND=$R/data/holdout/holdout464_vote45_band.csv
for CFG in "v45_a100:1:0" "v45_r1:1:1" "v45_nc:1:1"; do
  NAME=${CFG%%:*}; REST=${CFG#*:}; NC=${REST%%:*}; RP=${REST##*:}
  o=$R/outputs/tir_${NAME}_holdout_vote45.jsonl
  [ -f "$o" ] && { say "SKIP $NAME"; continue; }
  say "$NAME 생성 (nocode-retries=$NC repair-rounds=$RP)"
  $V $R/scripts/tir_repair_client.py \
    --input "$BAND" --output "$o" \
    --model hybrid3145 --num-samples 8 \
    --repair-rounds "$RP" --nocode-retries "$NC" \
    --exec-timeout 60 --exec-workers "$WORKERS" --request-workers 48 \
    --seed "2026092${NAME: -1}" \
    > $R/logs/tir_${NAME}_vote45.log 2>&1 || { say "$NAME 실패"; continue; }
  tail -2 $R/logs/tir_${NAME}_vote45.log
done

say "=== 2b) 표수4~5 스윕 판정 ==="
python3 $R/scripts/sweep_vote45_holdout.py 2>&1 | tee $R/logs/vote45_sweep.log

# ── 3) 32B teacher 생성 (밤새) ─────────────────────────────────────
say "=== 3) Qwen2.5-32B-Instruct teacher 생성 ==="
bash $R/scripts/run_teacher_32b.sh 2>&1 | tee $R/logs/teacher_32b.log

say "=== A100 야간 연쇄 완료 ==="
