#!/usr/bin/env bash
# NC 풀 생성이 끝나면 자기증류 continuation 을 이어서 돌린다.
#
# 순서 이유: NC 풀은 **검증된** 방향(8->16->24 매번 +1~5)이라 내일 제출 탄약으로 먼저
# 확보한다. 자기증류는 성공 확률 ~20% 의 도박이라 뒤로 민다.
# 32GB 에서 서버와 학습은 공존 불가(CONTEXT 22절)라 NC(서버 up) -> 학습(서버 down) 직렬.
#
# ⚠️ 종료는 PID 로만. pkill -f 는 자기 명령줄을 매칭해 자신을 죽인다(14절, 세 번 밟음).
set -u
R=/workspace/DLC
say(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*"; }
up(){ curl -s -m 3 -o /dev/null http://localhost:8000/v1/models; }

# ── 1) NC 풀 완료 대기 ──
say "=== 1) NC 풀 완료 대기 ==="
while pgrep -f "[r]un_nc_pools.sh" > /dev/null; do sleep 60; done
for S in 20260904 20260905; do
  f=$R/outputs/tir_nc_831_gate282_seed${S}.jsonl
  [ -f "$f" ] && say "NC seed $S: $(wc -l < "$f")행" || say "NC seed $S: 없음(실패)"
done

# ── 2) 서버 내리고 자기증류 학습 (batch 2x8 — 8x2 는 32GB 에서 OOM) ──
say "=== 2) 자기증류 continuation 학습 ==="
pids=$(pgrep -f "vllm.entrypoints.openai.api_server" || true)
[ -n "$pids" ] && { say "서버 종료 (PID: $pids)"; kill $pids; sleep 20; }

if [ ! -f "$R/checkpoints/tir_selfdistill_r8qv_lr5e7_e1/final_adapter/adapter_config.json" ]; then
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True python3 $R/scripts/train_qlora.py \
    --data $R/data/processed/tir_selfdistill.jsonl \
    --init-adapter $R/checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter \
    --output-dir $R/checkpoints/tir_selfdistill_r8qv_lr5e7_e1 \
    --learning-rate 5e-7 --epochs 1 \
    --batch-size 2 --gradient-accumulation 8 \
    --max-seq-length 2048 --max-minutes 90 \
    --group-by-length --save-steps 5000 --seed 20260902 \
    > $R/logs/selfdistill_train3.log 2>&1 || { say "학습 실패"; tail -20 $R/logs/selfdistill_train3.log; exit 1; }
fi
say "학습 완료"
grep -oE '"train_loss": [0-9.]+' $R/logs/selfdistill_train3.log | tail -1

# ── 3) 서버 재기동 + 어댑터 핫로드 ──
say "=== 3) 서버 재기동 ==="
if ! up; then
  VLLM_GPU_FRAC=0.90 setsid nohup bash $R/scripts/vllm_server.sh > $R/logs/vllm.log 2>&1 < /dev/null &
  for _ in $(seq 1 120); do sleep 15; up && break; done
fi
up || { say "서버 기동 실패"; exit 1; }
curl -s -m 60 -X POST http://localhost:8000/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"selfdistill\",\"lora_path\":\"$R/checkpoints/tir_selfdistill_r8qv_lr5e7_e1/final_adapter\"}" \
  -w " (load_lora http=%{http_code})\n"

# ── 4) 홀드아웃87 평가 — 신규 vs 기준, 같은 시드/설정. 변수는 어댑터 하나뿐 ──
say "=== 4) 홀드아웃87 평가 ==="
for M in selfdistill hybrid3145; do
  o=$R/outputs/selfdistill_eval_${M}_holdout87.jsonl
  [ -f "$o" ] && { say "SKIP $M"; continue; }
  /workspace/venv-vllm/bin/python $R/scripts/tir_repair_client.py \
    --input $R/data/holdout/holdout464_vote3.csv --output "$o" \
    --model "$M" --num-samples 8 --repair-rounds 1 --nocode-retries 1 \
    --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed 20260907 \
    > $R/logs/selfdistill_eval_${M}.log 2>&1 || { say "평가 $M 실패"; continue; }
  say "$M: $(tail -1 $R/logs/selfdistill_eval_${M}.log)"
done

python3 $R/scripts/report_selfdistill_gate.py \
  --new $R/outputs/selfdistill_eval_selfdistill_holdout87.jsonl \
  --base $R/outputs/selfdistill_eval_hybrid3145_holdout87.jsonl \
  --new-log $R/logs/selfdistill_eval_selfdistill.log \
  --base-log $R/logs/selfdistill_eval_hybrid3145.log 2>&1 | tee $R/logs/selfdistill_gate.log
say "=== 전부 완료 ==="
