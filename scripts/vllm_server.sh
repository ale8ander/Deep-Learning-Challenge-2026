#!/usr/bin/env bash
# vLLM 상주 서버. 매 실험마다 8분씩 드는 기동 비용을 한 번으로 끝낸다.
#
# 왜 필요한가: /workspace가 네트워크 마운트(mfs)라 vLLM import에만 ~6분,
# 모델 로딩·컴파일에 ~2분이 든다. 실험 한 번이 10분인데 실제 생성은 2.5분뿐이다.
# 서버를 띄워두면 실험이 생성 시간만 남는다.
#
# GPU 메모리: 0.45로 잡아 학습(batch 8 기준 ~20GB)과 공존할 수 있게 한다.
# 서버만 쓸 때 더 큰 KV 캐시가 필요하면 VLLM_GPU_FRAC=0.85 로 올린다.
set -u
cd "$(dirname "$0")/.."
mkdir -p logs

PORT=${VLLM_PORT:-8000}
FRAC=${VLLM_GPU_FRAC:-0.45}

# LoRA 어댑터를 이름으로 등록해두면 요청마다 model= 로 골라 쓸 수 있다.
# 같은 서버에서 여러 계보를 비교할 수 있다는 뜻이다.
# run_inference.sh 가 쓰는 5종 전부를 등록해야 한다 (voter: hybrid3145/h3244/ext3000/h4145,
# 게이트: ck150). verify voter 는 hybrid3145 가중치에 프롬프트만 다르다.
ADAPTERS=(
  "hybrid3145=checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter"
  "h3244=checkpoints/hybrid_3244_r8_qv_lr2e6_e1/final_adapter"
  "ext3000=checkpoints/external_3000_r8_qv_lr2e6_e1/final_adapter"
  "h4145=checkpoints/hybrid_4145_r8_qv_lr1p5e6_e1/final_adapter"
  "ck150=checkpoints/grpo_3145_scaleup_r8_qv_lr2e6_steps800_g8/checkpoint-150"
)
LORA_ARGS=()
for a in "${ADAPTERS[@]}"; do
  name="${a%%=*}"; path="${a#*=}"
  if [ -d "$path" ]; then
    LORA_ARGS+=("${name}=${path}")
  else
    echo "[경고] 어댑터 없음, 건너뜀: ${name} (${path})"
  fi
done

echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] vLLM 서버 기동 (port=${PORT}, gpu=${FRAC})"
echo "  등록 어댑터: ${LORA_ARGS[*]}"

# RTX 5090(sm_120)에서 FlashInfer JIT 이 "requires GPUs with sm75 or higher"로 죽는다.
# 실제로는 arch 파싱이 12.0을 못 다루는 것이고, 네이티브 샘플러로 내리면 정상 동작한다.
# A100/Blackwell 96GB pod 에서는 이 변수 없이도 떴다 — 5090 전용 우회다.
# VLLM_ALLOW_RUNTIME_LORA_UPDATING: 어댑터 추가할 때마다 재기동 8분을 날리지 않기 위함.
exec env PATH=/workspace/venv-vllm/bin:$PATH PYTHONUNBUFFERED=1 \
  VLLM_USE_FLASHINFER_SAMPLER=0 \
  VLLM_ALLOW_RUNTIME_LORA_UPDATING=1 \
  /workspace/venv-vllm/bin/python -m vllm.entrypoints.openai.api_server \
  --model /workspace/models/Qwen2.5-3B-Instruct \
  --served-model-name base \
  --dtype float16 \
  --max-model-len "${MAXLEN:-8192}" \
  --gpu-memory-utilization "${FRAC}" \
  --enable-lora \
  --max-lora-rank 16 \
  --max-loras 11 \
  --lora-modules "${LORA_ARGS[@]}" \
  --port "${PORT}" \
  --no-enable-log-requests
