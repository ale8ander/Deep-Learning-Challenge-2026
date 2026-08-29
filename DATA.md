# 사용 데이터 목록

최종 제출본(665/831)을 만든 모델들이 사용한 데이터 전체입니다.
각 파일의 정확한 구성·sha256·오염 제거 기록은 `data/**/**.manifest.json` 에 기계 판독
가능한 형태로 보존돼 있습니다.

## 1. 대회 공식 데이터

| 데이터 | 용도 |
|---|---|
| 공식 train (17,000문제) | SFT 소재 선별, GRPO 학습 풀(667문제), 홀드아웃 검증셋 |
| 공식 test (`test.parquet` / 리더보드 831) | **학습에 사용하지 않음** (추론 대상만) |

## 2. 외부 공개 데이터 (전부 무료·공개, 라이선스 명시)

| 데이터셋 | 리비전 | 라이선스 | 사용처 |
|---|---|---|---|
| [AI-MO/NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) | `1b05109f` | Apache-2.0 | external_3000 (일부), hybrid_3145 내 300, hybrid_4145 추가분 1,000 |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `740312a` | MIT | external_3000 (1,000) |
| [DigitalLearningGmbH/MATH-lighteval](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval) | `92ace7ed` | MIT | external_3000 (1,200), hybrid_3145 내 700 |

선별은 전부 결정적(sha256 기반 고정 시드 샘플링, `scripts/prepare_hard_math_sft.py`)이며,
공식 train 17,000 + 리더보드 831 문제와의 중복은 정규화 exact match + token-Jaccard /
SequenceMatcher 근사 중복 검사로 제거했습니다 (manifest 의 `official_decontamination` 항목).

## 3. 상용 API 생성 CoT (학습 데이터 구축 목적 — 규정상 허용)

`hybrid_3145.jsonl` 의 풀이(CoT) 중 2,145개는 상용 API 로 생성했습니다.
**학습 데이터 구축에만 사용**했으며, test 문제를 상용 API 에 입력하거나 추론 시점에
외부 모델을 호출한 사실은 없습니다.

| teacher | 샘플 수 |
|---|---|
| gpt-5.6-luna | 1,906 |
| gpt-5.4-mini-2026-03-17 | 133 |
| gpt-5.6-luna-high | 82 |
| gpt-5.6-terra | 19 |
| gpt-5.4-mini-high | 5 |

나머지는 사람이 쓴 공개 풀이(MATH-lighteval 700, NuminaMath 큐레이션 300)입니다.
문항 원문은 공식 train 및 위 공개 데이터셋에서만 가져왔습니다.

## 4. GRPO (RLVR)

`data/processed/grpo_passrate_scaleup.jsonl` — 공식 train 에서 hybrid_3145 의
stochastic pass-rate 2~6/8 대역 667문제를 선별. 보상은 **공식 정답 라벨과의 exact match**
만 사용(타 모델의 채점·판정 개입 없음).

## 5. 실험만 하고 최종 제출에는 미사용

다음 데이터는 실험 후 기각되어 최종 제출 경로에 포함되지 않습니다 (저장소에 스크립트/
어댑터가 남아 있는 것은 과정 투명성을 위한 보존입니다): NuminaMath-TIR 증류,
DeepSeek-R1 공개 CoT 셋, Qwen2.5-32B 생성 증류, 자기증류 각종, external_10000 확장본.

## 6. 비고

- `data/processed/hybrid_4145.jsonl` 은 실패 실험 정리 시 삭제되어 저장소에 없습니다.
  sha256 `70f2053e978e6e0a3d3ca8483461b06779170aff1b45424667dbd93e9ddcb33b`
  (해당 어댑터 `training_metadata.json` 에 기록)이며, 구성은 hybrid_3145.jsonl 전체 +
  NuminaMath-1.5 결정적 샘플링 1,000문제입니다.
- TIR(코드 실행)은 데이터가 아니라 추론 기법이며, 실행은 전부 로컬 파이썬
  서브프로세스입니다. sympy(BSD) 등 표준 공개 라이브러리만 사용합니다.
