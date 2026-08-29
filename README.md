# 아주 소중한 딥러닝 챌린지 2026

Qwen2.5-3B-Instruct 단일 베이스 규정에서 LoRA 어댑터 앙상블 + test-time 파이프라인으로
Public 리더보드 **0.80024 (665/831)** 를 기록한 제출물입니다.

- 최종 제출: `submissions/submission_ck150_gate5_sup4_codeguard.csv`
- 백업(코드 실행 미사용본): `submissions/submission_self_consistency_hybrid3145_n8_min4_support4.csv`

## 방법 요약

베이스는 전 구간 Qwen2.5-3B-Instruct 하나이고, 데이터를 달리해 학습한 LoRA(r8, q/v) 어댑터들을
추론 시점에 조합합니다.

1. 5-voter 다수결 — 어댑터 4종 greedy + hybrid_3145에 self-verification 프롬프트를 얹은 것 1종
2. 4/5 합의 문항만 hybrid_3145 stochastic N=8 다수결로 교체 (self-consistency)
3. 표가 갈리는 문항은 TIR로 교체 — 모델이 쓴 파이썬을 로컬 서브프로세스로 실행해 검증한 답의 다수결
4. 삼중 게이트 — support≤4 문항에서 GRPO 체크포인트의 N=8이 5표 이상 확신하고 코드가 반대하지 않으면 교체

추론은 전부 로컬(vLLM/HF)이고 외부 API·검색은 쓰지 않았습니다. `test.parquet`은 학습에 사용하지 않았습니다.

## 재현

```bash
# 조립 검증: 보관된 추론 산출물 -> 제출 CSV, 바이트 단위 대조 (GPU 불필요, ~1분)
bash scripts/reproduce_all.sh compose

# 전체 재생성: 생성부터 다시 (GPU, ~2시간. 샘플링이 확률적이라 바이트 일치는 안 됨)
bash scripts/reproduce_all.sh full
```

compose는 656 → 660 → 665 → 664 제출본 5단계를 전부 831행 일치로 검증합니다.

학습 재현:

```bash
# SFT (voter 4종 공통, 예: hybrid_3145)
python3 scripts/train_qlora.py --data data/processed/hybrid_3145.jsonl \
  --learning-rate 2e-6 --lora-rank 8 --epochs 1 --seed 2026

# GRPO (ck150): hybrid_3145에서 이어 학습, 곡선 정점 checkpoint-150 채택
python3 scripts/train_grpo_qlora.py --data data/processed/grpo_passrate_scaleup.jsonl \
  --adapter-path checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter \
  --output-dir checkpoints/grpo_3145_scaleup_r8_qv_lr2e6_steps800_g8 \
  --max-steps 800 --learning-rate 2e-6 --batch-size 8 --gradient-accumulation 2 \
  --num-generations 8 --max-completion-length 512 --beta 0.005 \
  --save-steps 50 --seed 20260917
```

## 체크포인트

| 어댑터 | 데이터 | LR | seed |
|---|---|---|---|
| hybrid_3145_r8_qv_lr2e6_e1 | hybrid_3145.jsonl (3,145) | 2e-6 | 2026 |
| hybrid_3244_r8_qv_lr2e6_e1 | hybrid_3244.jsonl (3,244) | 2e-6 | 2026 |
| external_3000_r8_qv_lr2e6_e1 | external_math_3000.jsonl (3,000) | 2e-6 | 2026 |
| hybrid_4145_r8_qv_lr1p5e6_e1 | hybrid_4145.jsonl (4,145)* | 1.5e-6 | 2026 |
| grpo_.../checkpoint-150 | grpo_passrate_scaleup.jsonl (667) | 2e-6 | 20260917 |

하이퍼파라미터·데이터 sha256 전체는 각 디렉터리의 `training_metadata.json` 참고.
verify voter는 별도 가중치가 아니라 hybrid_3145 + 검증 프롬프트입니다.

\* hybrid_4145.jsonl은 실험 정리 중 삭제됨. sha256은 metadata에 있고, 구성은
hybrid_3145.jsonl + NuminaMath-1.5 결정적 샘플링 1,000문제 (생성 스크립트 포함).

## 환경

torch 버전 충돌 때문에 환경이 둘로 나뉩니다.

- `requirements.txt` — 학습/평가/조립 (torch 2.8, transformers 4.57)
- `requirements-vllm.txt` — vLLM 추론 서버용 별도 venv (torch 2.13)

베이스 모델은 `/workspace/models/Qwen2.5-3B-Instruct`, 리더보드 test CSV는
`data/deep_chal_math_leaderboard_filtered.csv` 경로에 두면 됩니다 (둘 다 저장소 미포함).
스크립트가 `/workspace/DLC` 절대 경로를 쓰므로 이 위치에 클론하는 걸 권장합니다.
확인된 GPU: RTX 5090 32GB, A100 80GB.

## 구성

```
checkpoints/   LoRA 어댑터 5종 + training_metadata.json
data/          학습 데이터 + manifest (출처/sha256/오염 제거 기록)
outputs/       compose 재현용 추론 산출물
scripts/       학습·추론·조립 스크립트
submissions/   제출 CSV (최종본 + 재현 검증 대상)
logs/          GRPO 학습 로그
DATA.md        사용 데이터 목록
REPORT.md      실험 보고서
```
