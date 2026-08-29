# DLC 수학 추론 대회 — 제출물 (검증용)

베이스 모델 **Qwen/Qwen2.5-3B-Instruct** 단일 고정 규정 하에서,
LoRA SFT 4종 + GRPO(RLVR) 1종의 어댑터 앙상블과 규칙 기반 test-time 파이프라인으로
**Public 리더보드 0.80024 (665/831)** 를 기록한 프로젝트입니다.

- **최종 제출**: `submission_ck150_gate5_sup4_codeguard.csv` (665/831)
- **백업 제출**: `submission_self_consistency_hybrid3145_n8_min4_support4.csv` (623/831, 코드 실행(TIR) 미사용본)

## 1. 아키텍처 개요

베이스는 전 구간 Qwen2.5-3B-Instruct 하나이며, 서로 다른 데이터로 학습한
LoRA 어댑터(r8, q/v proj)들을 추론 시점에 조합합니다. 파이프라인은 4층입니다.

```
[1층] 5-voter 다수결
      hybrid_3145 / hybrid_3244 / external_3000 / hybrid_4145 (greedy)
      + hybrid_3145 가중치에 self-verification 프롬프트(verify)  = 5표
      → 유일 최빈값 ≥2표 채택, 아니면 hybrid_3145 답
[2층] support=4 self-consistency override
      정확히 4/5 합의 문항만 hybrid_3145 stochastic N=8 (temp 0.7) 의
      유일 최빈값 ≥4표로 교체
[3층] TIR(코드 실행) override — 전부 로컬 파이썬 서브프로세스, 외부 API 없음
      · SC 표수≤3 문항: TIR 샘플 3풀(24샘플)의 코드검증 답 min-count 2
      · SC 표수 4~5 & risky≥1 문항: TIR 16샘플 min-count 2
[4층] 삼중 게이트 (최종 +5)
      5-voter support≤4  ×  GRPO ck150 N=8 유일최빈 ≥5표
      ×  코드가드(TIR 검증 표가 기존 답을 지지하면 교체 취소)
```

규정 관련: 추론은 전부 로컬(vLLM/HF), 외부 API·검색 미사용. TIR 의 코드 실행은
모델이 생성한 파이썬을 로컬 서브프로세스로 실행해 정수 출력을 대조하는 것입니다.
`test.parquet` 은 어떤 학습에도 사용하지 않았습니다.

## 2. 최종 모델 체크포인트 (`checkpoints/`)

전부 Qwen2.5-3B-Instruct 위의 LoRA (rank 8, target q_proj/v_proj). 상세 하이퍼파라미터와
데이터 sha256 은 각 디렉터리의 `training_metadata.json` 에 있습니다.

| 어댑터 | 학습 데이터 | 방식 | LR | seed |
|---|---|---|---|---|
| `hybrid_3145_r8_qv_lr2e6_e1` | `data/processed/hybrid_3145.jsonl` (3,145) | SFT 1ep | 2e-6 | 2026 |
| `hybrid_3244_r8_qv_lr2e6_e1` | `data/processed/hybrid_3244.jsonl` (3,244) | SFT 1ep | 2e-6 | 2026 |
| `external_3000_r8_qv_lr2e6_e1` | `data/external/external_math_3000.jsonl` (3,000) | SFT 1ep | 2e-6 | 2026 |
| `hybrid_4145_r8_qv_lr1p5e6_e1` | hybrid_4145.jsonl (4,145) ※아래 참고 | SFT 1ep | 1.5e-6 | 2026 |
| `grpo_3145_scaleup_.../checkpoint-150` | `data/processed/grpo_passrate_scaleup.jsonl` (667문제) | GRPO(RLVR), hybrid_3145 에서 이어 학습 | 2e-6 | 20260917 |

- verify voter 는 별도 가중치가 아니라 **hybrid_3145 어댑터 + self-verification 시스템 프롬프트**입니다 (`scripts/submit_baseline.py` 의 `SYSTEM_PROMPTS["verify"]`).
- ※ `hybrid_4145.jsonl` 은 실패 실험 정리 과정에서 삭제됐습니다.
  구성은 hybrid_3145.jsonl + NuminaMath-1.5 결정적 샘플링 1,000문제이고,
  파일 sha256(`70f2053e…`)이 `training_metadata.json` 에 보존돼 있으며
  생성 스크립트(`scripts/prepare_hard_math_sft.py`, `scripts/merge_sft_jsonl.py`)가 포함돼 있습니다.

## 3. 재현 (submission.csv)

### 3-1. compose — 결정론적 조립 검증 (권장, GPU 불필요, ~1분)

보관된 추론 산출물(`outputs/*.jsonl`, 이 저장소에 포함)에서 최종 CSV 를 조립해
제출본과 **바이트 단위로** 대조합니다.

```bash
bash scripts/reproduce_all.sh compose
```

검증 항목: 챔피언 체인 → 656 본 → 660 본 → **665 최종본** → 664 게이트v3 본, 총 5단계
전부 831행 일치를 확인합니다.

### 3-2. full — 빈 상태에서 전체 재생성 (GPU, ~2시간)

```bash
bash scripts/reproduce_all.sh full --smoke 20   # 20문제 스모크
bash scripts/reproduce_all.sh full              # 전체
```

voter greedy → SC N=8 → TIR 풀 → 조립 순서로 새로 생성합니다. 샘플링이 확률적이라
(같은 시드라도 GPU/엔진에 따라 커널이 달라) 바이트 일치는 원리적으로 보장되지 않으며,
규칙·절차의 재현을 확인하는 용도입니다.

### 3-3. 학습 재현

```bash
# SFT (voter 4종, 예: hybrid_3145)
python3 scripts/train_qlora.py --data data/processed/hybrid_3145.jsonl \
  --learning-rate 2e-6 --lora-rank 8 --epochs 1 --seed 2026

# GRPO (ck150): hybrid_3145 에서 이어 학습, step 150 체크포인트 채택
python3 scripts/train_grpo_qlora.py --data data/processed/grpo_passrate_scaleup.jsonl \
  --adapter-path checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter \
  --output-dir checkpoints/grpo_3145_scaleup_r8_qv_lr2e6_steps800_g8 \
  --max-steps 800 --learning-rate 2e-6 --batch-size 8 --gradient-accumulation 2 \
  --num-generations 8 --max-completion-length 512 --beta 0.005 \
  --save-steps 50 --seed 20260917
# (실제 학습은 사전 등록한 조기 종료 규칙에 따라 step ~250 에서 중단, 곡선 정점 ck150 채택)
```

GRPO 학습 풀(667문제)은 공식 train 에서 stochastic pass-rate 2~6/8 대역을 선별한 것:
`scripts/screen_grpo_passrate.py` → `scripts/build_grpo_passrate_pool.py`
(리더보드 831 문제는 스크리닝·학습에 사용하지 않음). 학습 로그: `logs/grpo_scaleup_train.log`.

## 4. 실행 환경

환경은 두 개입니다 (torch 버전 충돌로 분리, 각 파일 머리말 참고):

- `requirements.txt` — 학습/평가/조립 (torch 2.8 + cu128, transformers 4.57)
- `requirements-vllm.txt` — vLLM 추론 서버 전용 venv (torch 2.13 + cu130)

```bash
python3 -m pip install -r requirements.txt
python3 -m venv /workspace/venv-vllm && /workspace/venv-vllm/bin/pip install -r requirements-vllm.txt
```

- 베이스 모델은 `/workspace/models/Qwen2.5-3B-Instruct` 에 HF 원본을 받아둡니다.
- 리더보드 test CSV 는 `data/deep_chal_math_leaderboard_filtered.csv` 경로에 둡니다 (저장소 미포함).
- 스크립트가 `/workspace/DLC` 절대 경로를 참조하므로 해당 경로에 클론하는 것을 권장합니다.
- 검증된 GPU: RTX 5090 32GB / A100 80GB (추론 서버: `bash scripts/vllm_server.sh`).

## 5. 저장소 구성

```
checkpoints/   최종 LoRA 어댑터 5종 (+ training_metadata.json)
data/          학습 데이터(최종 사용분) + manifest (출처·sha256·오염 제거 기록)
outputs/       compose 재현에 필요한 추론 산출물 (voter/SC/TIR/게이트 신호)
scripts/       학습·추론·조립·분석 스크립트 전체 (실패 실험 포함, 과정 투명성)
logs/          GRPO 학습 로그
submissions/  전체 제출 이력 (최종본 포함)
DATA.md        사용 데이터 목록 (필수 제출 항목)
REPORT.md      실험 보고서 (접근 방법·전략)
```

핵심 스크립트만 추리면: `reproduce_all.sh`(재현 진입점) · `rebuild_chain.py`(1~3층 조립) ·
`build_merged16_submission.py`(TIR 병합) · `build_ck_gate_submission.py`(4층 게이트) ·
`train_qlora.py`/`train_grpo_qlora.py`(학습) · `tir_repair_client.py`(TIR 하네스) ·
`gen_client.py`/`evaluate_self_consistency.py`(생성).
