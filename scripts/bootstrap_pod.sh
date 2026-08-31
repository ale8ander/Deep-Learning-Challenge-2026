#!/usr/bin/env bash
# pod 스왑 직후 1회 실행. 환경 복구 + 상태 점검.
#
# /workspace 는 볼륨이라 보존된다 — 모델·체크포인트·venv-vllm·데이터·스크립트 전부 살아있다.
# 날아가는 건 **시스템 python3 패키지뿐**이다(학습용 torch/transformers/peft...).
#
# 사용: bash scripts/bootstrap_pod.sh
set -u
R=$(cd "$(dirname "$0")/.." && pwd)
say(){ echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] $*"; }

say "=== 1) 하드웨어 ==="
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
echo "  CPU 코어: $(nproc)   RAM: $(free -g | awk '/^Mem:/{print $2}')GB"
VRAM=$(nvidia-smi --query-gpu=memory.total --format=csv,noheader,nounits | head -1)

say "=== 2) 볼륨 보존 확인 ==="
for p in /workspace/models/Qwen2.5-3B-Instruct /workspace/venv-vllm/bin/python \
         $R/checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter \
         $R/submissions/submission_pool24_v3mc2.csv $R/data/processed/r1_distill_3k.jsonl \
         $R/data/processed/distill_target_pool.jsonl; do
  [ -e "$p" ] && echo "  OK   $p" || echo "  !! 없음 $p"
done

say "=== 3) 시스템 python 의존성 복구 ==="
if python3 -c "import torch, transformers, peft, trl" 2>/dev/null; then
  say "이미 설치됨 — 건너뜀"
else
  python3 -m pip install --no-cache-dir -q -r $R/requirements.txt \
    && say "설치 완료" || { say "설치 실패"; exit 1; }
fi
python3 -c "
import torch, transformers, peft
print(f'  torch {torch.__version__} | CUDA {torch.version.cuda} | GPU {torch.cuda.is_available()}')
print(f'  transformers {transformers.__version__} | peft {peft.__version__}')"

say "=== 4) vLLM 환경 (볼륨에 보존됨) ==="
/workspace/venv-vllm/bin/python -c "
import vllm, torch
print(f'  vllm {vllm.__version__} | torch {torch.__version__}')" 2>/dev/null || say "!! venv-vllm 손상"

say "=== 5) GPU 별 권장 설정 ==="
if [ "$VRAM" -gt 60000 ]; then
  echo "  80GB급 감지 —"
  echo "    · 서버 VLLM_GPU_FRAC=0.45 로 띄우면 **학습과 공존** (재기동 13분 세금 소멸)"
  echo "    · 32B AWQ teacher 생성 가능 (~19GB)"
  echo "    · exec-workers 를 코어 수에 맞춰 올릴 것 (5090 의 32 가 병목이었다)"
else
  echo "  32GB급 — 서버(0.90)와 학습은 공존 불가. 순차 실행 필요"
fi

say "=== 6) 진행 중이던 작업 상태 ==="
[ -f "$R/checkpoints/r1_distill_r8qv_lr1e5/final_adapter/adapter_config.json" ] \
  && echo "  R1 증류 어댑터: 있음(학습 완료됨)" \
  || echo "  R1 증류 어댑터: 없음 → 재학습 필요"
echo "  확보된 제출본: $(ls $R/submissions/submission_pool24_v3mc2.csv 2>/dev/null && echo '' || echo '없음')"

say "=== 부트스트랩 완료 ==="
