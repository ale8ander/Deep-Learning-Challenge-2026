#!/usr/bin/env bash
# teacher 32B 증류 판정 체인 — A100 용. 데이터(teacher32b_sft_full.jsonl)는 B200 에서 생성 완료됨.
# 실행: setsid nohup bash scripts/run_teacher_judge_a100.sh > logs/teacher_judge.log 2>&1 < /dev/null &
set -u
R=/workspace/DLC
V=/workspace/venv-vllm/bin/python
CK=$R/checkpoints/teacher32b_r8qv_lr2e6
D=$R/data/processed/teacher32b_sft_full.jsonl
say(){ echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] $*"; }
up(){ curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q hybrid3145; }
cd $R

[ -s "$D" ] || { say "SFT 데이터 없음: $D"; exit 1; }
say "SFT 데이터 $(wc -l < $D)건"

say "=== 1) 학습: hybrid_3145 이어서, r8 qv LR 2e-6 1ep (hybrid_3145 를 만든 검증 레시피) ==="
if [ ! -f "$CK/final_adapter/adapter_config.json" ]; then
  env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True /usr/bin/python3 scripts/train_qlora.py \
    --data "$D" --output-dir "$CK" \
    --init-adapter checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter \
    --learning-rate 2e-6 --epochs 1 \
    --batch-size 8 --gradient-accumulation 2 \
    --max-seq-length 3072 --max-minutes 90 \
    --group-by-length --save-steps 5000 --seed 20260914 \
    > logs/teacher_sft_train.log 2>&1 || { say "학습 실패"; tail -25 logs/teacher_sft_train.log; exit 1; }
fi
say "학습 완료: $(grep -oE '"train_loss": [0-9.]+' logs/teacher_sft_train.log | tail -1)"

say "=== 2) 서버 확인/기동 + 핫로드 ==="
if ! up; then
  VLLM_GPU_FRAC=0.90 setsid nohup bash scripts/vllm_server.sh > logs/vllm.log 2>&1 < /dev/null &
  for _ in $(seq 1 120); do sleep 15; up && break; done
fi
up || { say "서버 기동 실패"; exit 1; }
curl -s -X POST http://localhost:8000/v1/load_lora_adapter \
  -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"teacher32b\",\"lora_path\":\"$CK/final_adapter\"}"
echo

say "=== 3) 게이트 A — 홀드아웃87 greedy (기준: hybrid3145 = 18, 8/29 새벽 실측) ==="
$V scripts/gen_client.py \
  --input data/holdout/holdout464_vote3.csv \
  --output outputs/teachersft_gen_holdout87.jsonl \
  --model teacher32b --max-new-tokens 2048 --request-workers 87 2>&1 | tail -2

say "=== 4) 게이트 B (본선) — TIR 하네스 8샘플 (기준: 챔피언 18/87) ==="
$V scripts/tir_repair_client.py \
  --input data/holdout/holdout464_vote3.csv \
  --output outputs/teachersft_tir8_holdout87.jsonl \
  --model teacher32b --num-samples 8 --repair-rounds 1 --nocode-retries 1 \
  --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed 20260915 2>&1 | tail -2

say "=== 5) 판정 ==="
/usr/bin/python3 - <<'PY'
import json, sys
sys.path.insert(0,"/workspace/DLC/scripts")
from tir_common import normalize as n
def load(p): return [json.loads(l) for l in open(p) if l.strip()]
sc=lambda rows: sum(1 for r in rows if n(r.get("prediction")) is not None and n(r.get("prediction"))==n(r.get("answer")))
base=load("/workspace/DLC/outputs/r1distill_gen_hybrid3145_holdout87.jsonl")
new=load("/workspace/DLC/outputs/teachersft_gen_holdout87.jsonl")
print(f"[게이트A greedy]  hybrid3145 {sc(base)}/87  ->  teacherSFT {sc(new)}/87  ({sc(new)-sc(base):+d})")
tir=load("/workspace/DLC/outputs/teachersft_tir8_holdout87.jsonl")
tc=sc(tir)
print(f"[게이트B TIR8]   챔피언 18/87      ->  teacherSFT {tc}/87  ({tc-18:+d})")
print()
print("판정 규칙: 87문제 SE ±3.5~3.9 (C-3). 게이트B +4 이상일 때만 채택 논의.")
print("채택 경로 = base 교체 (풀 재생성 ~1.5h). 새 계보 풀 추가는 7연패라 금지.")
PY
say "=== 판정 체인 완료 ==="
