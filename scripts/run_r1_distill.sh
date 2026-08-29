#!/usr/bin/env bash
# R1 long-CoT 증류 스모크 — **미개척 변수 하나만** 바꾼다: 응답 길이.
#
# 기존 SFT 전부: assistant 평균 436자(hybrid_3145) / 706자(verbose)
# 이번:          assistant 1,200~6,000자  = 5~14배
# 18절의 "흡수 벽"은 전부 짧은 CoT 실험이었으므로 이 칸은 아직 안 닫혔다.
#
# 판정도 지표가 다르다. 코드 생성률이 아니라 **응답 길이**가 1차 흡수 지표다.
# 길이가 안 늘면 학습이 안 된 것이고, 길이만 늘고 점수가 떨어지면 22절 패턴이다.
set -u
R=/workspace/DLC
V=/workspace/venv-vllm/bin/python
CK=$R/checkpoints/r1_distill_r8qv_lr1e5
say(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*"; }
up(){ curl -s -m 3 -o /dev/null http://localhost:8000/v1/models; }

say "=== 1) 데이터 대기 ==="
while pgrep -f "[b]uild_r1_distill_sft" >/dev/null; do sleep 30; done
D=$R/data/processed/r1_distill_3k.jsonl
[ -s "$D" ] || { say "데이터 없음"; exit 1; }
say "데이터 $(wc -l < $D)건"

say "=== 2) 서버 내리고 학습 ==="
pids=$(pgrep -f "vllm.entrypoints.openai.api_server" || true)
[ -n "$pids" ] && { say "서버 종료 (PID: $pids)"; kill $pids; sleep 20; }

# max_seq 3072: 6000자 ≈ 1800토큰 + 문제 + 시스템. 32GB 에서 batch 1x16 로 맞춘다.
# LR 1e-5: 흡수를 실제로 일으켜야 하므로 자기증류(5e-7, 흡수 0)보다 훨씬 높게.
#          rank32/LR1e-5 는 -24 를 만든 설정이지만 그건 rank32 였다. r8 은 용량이 작다.
if [ ! -f "$CK/final_adapter/adapter_config.json" ]; then
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 $R/scripts/train_qlora.py \
    --data "$D" --output-dir "$CK" \
    --lora-rank 8 --target-modules qv \
    --learning-rate 1e-5 --epochs 1 \
    --batch-size 1 --gradient-accumulation 16 \
    --max-seq-length 3072 --max-minutes 100 \
    --group-by-length --save-steps 5000 --seed 20260910 \
    > $R/logs/r1_distill_train.log 2>&1 || { say "학습 실패"; tail -25 $R/logs/r1_distill_train.log; exit 1; }
fi
say "학습 완료"; grep -oE '"train_loss": [0-9.]+' $R/logs/r1_distill_train.log | tail -1

say "=== 3) 서버 재기동 + 핫로드 ==="
if ! up; then
  VLLM_GPU_FRAC=0.90 setsid nohup bash $R/scripts/vllm_server.sh > $R/logs/vllm.log 2>&1 < /dev/null &
  for _ in $(seq 1 120); do sleep 15; up && break; done
fi
up || { say "서버 기동 실패"; exit 1; }
curl -s -m 60 -X POST http://localhost:8000/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"r1distill\",\"lora_path\":\"$CK/final_adapter\"}" \
  -w " (load_lora http=%{http_code})\n"

say "=== 4) 홀드아웃87 평가 (일반 생성, TIR 아님) ==="
# long-CoT 는 코드가 아니라 추론으로 푼다. gen_client 로 재야 흡수가 보인다.
for M in r1distill hybrid3145; do
  o=$R/outputs/r1distill_gen_${M}_holdout87.jsonl
  [ -f "$o" ] && { say "SKIP $M"; continue; }
  $V $R/scripts/gen_client.py \
    --input $R/data/holdout/holdout464_vote3.csv --output "$o" \
    --model "$M" --max-new-tokens 2048 --retry-max-new-tokens 3072 \
    --request-workers 32 --seed 20260911 \
    > $R/logs/r1distill_gen_${M}.log 2>&1 || { say "평가 $M 실패"; continue; }
  say "$M 완료"
done

say "=== 5) 흡수 판정 ==="
python3 $R/scripts/report_r1_distill.py 2>&1 | tee $R/logs/r1_distill_gate.log
say "=== 완료 ==="
