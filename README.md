# 아주 소중한 딥러닝 챌린지 2026

![base](https://img.shields.io/badge/base-Qwen2.5--3B--Instruct-4c72b0)
![public](https://img.shields.io/badge/Public%20리더보드-672%2F831%20%3D%200.80866-e8590c)
![inference](https://img.shields.io/badge/추론-100%25%20로컬%2C%20외부%20API%20없음-2f9e44)

3B 단일 베이스 규정에서 출발해, 데이터를 달리 학습한 LoRA 어댑터들을 추론 시점에
조합하는 방식으로 Public 리더보드 **672/831 (0.80866)** 을 기록했습니다.

핵심 아이디어는 간단합니다. 이 규모의 모델은 한 번의 greedy로는 틀리는 문제가 많지만,
정답이 여러 샘플 어딘가에는 들어 있는 경우가 훨씬 많습니다. 그래서 점수의 대부분은
학습이 아니라 **"어느 샘플의 답을 믿을지"를 고르는 추론 체인**에서 나왔습니다.
다수결, self-consistency, 코드 실행 검증, 그리고 두 종류의 게이트를 5단계로 쌓았습니다.

이 저장소에는 최종 체크포인트, 학습과 추론 전체 파이프라인, 사용 데이터와 재현 절차가
들어 있습니다. 최종 제출 CSV는 주최측에 직접 전달합니다.

> 📄 실험 보고서는 [REPORT.md](REPORT.md)에 있습니다.

## 파이프라인

```mermaid
flowchart TB
    subgraph 학습["🎓 학습 (베이스 Qwen2.5-3B-Instruct 고정)"]
        direction LR
        D1["📚 SFT 데이터 4종<br/>hybrid_3145, 3244, 4145<br/>external_3000"]
        D2["🎯 GRPO 풀 667문제<br/>pass-rate 2~6/8 대역"]
        T1["LoRA SFT<br/>r8, q/v<br/><i>train_qlora.py</i>"]
        T2["GRPO (RLVR)<br/>보상 = 정답 exact match<br/><i>train_grpo_qlora.py</i>"]
        A1["🧩 어댑터 4종<br/>+ verify 프롬프트 voter"]
        A2["🏆 ck150 어댑터<br/>checkpoint-150"]
        D1 --> T1 --> A1
        D2 --> T2
        A1 -.hybrid_3145 에서 이어 학습.-> T2 --> A2
    end

    subgraph 추론["⚡ 추론 체인 (전부 로컬 vLLM/HF, 외부 API 없음)"]
        direction TB
        L1["<b>1단계: 5-voter 다수결</b><br/>어댑터 4종 greedy + verify 프롬프트"]
        L2["<b>2단계: Self-Consistency</b><br/>support≤4 문항만 N=8 다수결 교체"]
        L3["<b>3단계: TIR 코드 검증</b><br/>표 갈림 문항 → 모델이 쓴 파이썬을<br/>로컬 실행, 검증된 답의 다수결"]
        L4["<b>4단계: ck150 삼중 게이트</b><br/>5-voter 합의 약함 × N=8 확신 ≥5표 × 코드가드"]
        L5["<b>5단계: few-shot 포인터 게이트</b><br/>3-shot 포인터 × 독립 16샘플 상대다수<br/>× 재샘플 확인 ≥2표"]
        L1 --> L2 --> L3 --> L4 --> L5
    end

    A1 ==> L1
    A2 ==> L4
    A2 ==> L5
    L5 ==> CSV["📄 제출 CSV (id, answer)<br/><b>Public 리더보드 672/831 = 0.80866</b>"]

    classDef data fill:#dbeafe,stroke:#3b82f6,color:#1e3a5f
    classDef train fill:#ede9fe,stroke:#8b5cf6,color:#3b2a6e
    classDef adapter fill:#dcfce7,stroke:#22c55e,color:#14532d
    classDef l1 fill:#fefce8,stroke:#eab308,color:#713f12
    classDef l2 fill:#fef9c3,stroke:#eab308,color:#713f12
    classDef l3 fill:#fde68a,stroke:#f59e0b,color:#78350f
    classDef l4 fill:#fdba74,stroke:#f97316,color:#7c2d12
    classDef l5 fill:#f97316,stroke:#c2410c,color:#ffffff
    classDef final fill:#fee2e2,stroke:#ef4444,color:#7f1d1d

    class D1,D2 data
    class T1,T2 train
    class A1,A2 adapter
    class L1 l1
    class L2 l2
    class L3 l3
    class L4 l4
    class L5 l5
    class CSV final
```

각 단계가 하는 일:

| 단계 | 어떤 문항을 | 무엇으로 바꾸나 |
|---|---|---|
| 1 | 전체 | 어댑터 4종 greedy + 자기검증 프롬프트 voter, 5표 다수결 |
| 2 | 5-voter 합의가 약한 곳: support ≤ 4 (support = 최다 답이 받은 표 수, 5면 만장일치) | hybrid_3145 stochastic N=8 다수결 |
| 3 | 표가 갈리는 곳 | 모델이 쓴 파이썬을 로컬 실행, 출력이 검증된 답의 다수결 |
| 4 | support≤4 중 GRPO가 확신하는 곳 | ck150 의 8샘플에서 최다 답이 유일하고 5표 이상이면 그 답으로, 단 코드 실행 결과가 반대하면 취소 |
| 5 | few-shot greedy가 다른 답을 가리키는 곳 | 두 어댑터(ck150, hybrid_3145)의 16샘플에서 그 답이 현행 답보다 표가 많고, few-shot 으로 8번 다시 뽑아도 2번 이상 나올 때만 교체 |

뒤 단계일수록 발동 조건이 좁고 근거를 여러 개 요구합니다. 넓게 바꾸는 규칙은
검증셋에서만 좋아 보이고 실제로는 배신하는 일이 많아서, 조건의 교집합이 성립할 때만
답을 바꾸도록 설계했습니다. test 문제는 어떤 형태로도 학습에 쓰지 않았습니다.

## 재현

파이프라인은 `학습 → (어댑터) → 추론 생성 → (샘플 재료) → 조립 → 제출 CSV` 로 이어지고,
아래 세 절은 각각 다른 구간을 재현합니다. 앞 구간의 결과물(어댑터는 `checkpoints/`, 생성
샘플은 `outputs/`)이 저장소에 보존돼 있어 어느 구간부터든 독립적으로 시작할 수 있습니다.

- **최종 테스트 추론**: 어댑터로 test 2,000문제를 처음부터 푸는 전체 경로 (GPU, 2~4시간)
- **학습**: 데이터에서 어댑터를 다시 만드는 레시피 (GPU, 수 시간. 완성본이 동봉돼 있어 생략 가능)
- **제출본 재현 검증**: 보존된 생성 샘플에서 답 선택 규칙만 재실행해, 리더보드 제출본과 sha256 이 일치하는지 확인 (GPU 불필요, 1분)

### 사전 준비

파이썬 환경이 **둘**입니다. vLLM 0.28 이 torch 2.13 / transformers 5.x 를 요구하는데
학습과 조립 쪽은 torch 2.8 / transformers 4.x 에서 검증돼, 한 환경에 담을 수 없었습니다.
어떤 스크립트를 돌리느냐에 따라 설치할 파일이 다릅니다 (버전과 요건은 [환경](#환경) 절 참고):

| 하려는 것 | 환경 | 설치 |
|---|---|---|
| 제출 CSV 조립과 검증 (`compose_final_submissions.py`, `reproduce_all.sh compose`), 학습 (`train_*.py`), 데이터 준비 | 시스템 `python3` | `requirements.txt` |
| 추론 재료 생성 (`vllm_server.sh` 서버 + `run_inference.sh` 의 생성 단계) | venv `/workspace/venv-vllm` | `requirements-vllm.txt` |

직접 환경을 고를 일은 없습니다. 추론 스크립트가 venv 인터프리터를
`/workspace/venv-vllm/bin/python` 절대 경로로 호출하므로 activate 도 필요 없고,
venv 를 저 경로에 만들어 두기만 하면 됩니다. 다만 `run_inference.sh` 는 중간의 문항
분류 집계(`mega_bands.py`)를 시스템 python3 로 돌리므로 **전체 추론을 재현하려면 두 환경이
모두 필요**하고,
제출본 재현 검증(`compose`)만 할 때는 시스템 쪽만 있으면 됩니다 (GPU 도 불필요).

```bash
# 1) 저장소 클론 (스크립트들이 자기 위치에서 저장소 루트를 찾으므로 어디에 클론해도 된다)
git clone <repo-url> ~/Deep-Learning-Challenge-2026 && cd ~/Deep-Learning-Challenge-2026

# 2) 베이스 모델 배치 (HF Qwen/Qwen2.5-3B-Instruct, 저장소 미포함)
hf download Qwen/Qwen2.5-3B-Instruct --local-dir /workspace/models/Qwen2.5-3B-Instruct

# 3) 환경 설치: 위 표에서 필요한 쪽 (전체 재현이면 둘 다)
python3 -m pip install -r requirements.txt
python3 -m venv /workspace/venv-vllm \
  && /workspace/venv-vllm/bin/pip install -r requirements-vllm.txt

# 4) (선택) 하드웨어, 모델, 환경 일괄 점검
bash scripts/bootstrap_pod.sh
```

### 최종 테스트 추론

```bash
# 1) vLLM 서버 (어댑터 등록 포함)
bash scripts/vllm_server.sh &

# 2) 추론: 모델이 문제를 푸는 단계 (GPU, 2~4시간).
# 체인 1~5단계가 쓸 샘플을 전부 생성: voter 5종 → SC N=8 → TIR → 게이트 재료
# 입력/출력 기본값이 data/deep_chal_math_dataset_test.csv → outputs/final 이므로
# 그대로 실행하면 된다 (다른 입력은 MEGA_IN/MEGA_OUT 환경변수로 재지정)
SKIP_N64=1 bash scripts/run_inference.sh

# 3) 제출 CSV 조립: 공식 test 는 answer 값이 비어 있어 조립만 한다
# (answer 가 채워진 검증셋을 넣으면 층별 채점표도 함께 출력)
python3 scripts/compose_final_submissions.py \
  --input data/deep_chal_math_dataset_test.csv --materials outputs/final \
  --emit c672=submissions/final_c672.csv
```

> ⚠️ 2단계(생성)는 시드를 고정해 두었지만 GPU 기종, 엔진 버전, 요청 동시성에 따라 출력이
> 달라질 수 있습니다(greedy 포함). 3단계(조립)는 결정적이라 재료가 같으면 항상 같은 CSV 가
> 나옵니다. 아래 [제출본 재현 검증](#제출본-재현-검증) 참고.

### 학습

```bash
# SFT: voter 4종 공통 레시피, hybrid_3145 예시
python3 scripts/train_qlora.py --data data/processed/hybrid_3145.jsonl \
  --learning-rate 2e-6 --lora-rank 8 --epochs 1 --seed 2026

# GRPO: hybrid_3145 에서 이어 학습, 곡선 정점인 checkpoint-150 채택
python3 scripts/train_grpo_qlora.py --data data/processed/grpo_passrate_scaleup.jsonl \
  --adapter-path checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter \
  --output-dir checkpoints/grpo_3145_scaleup_r8_qv_lr2e6_steps800_g8 \
  --max-steps 800 --learning-rate 2e-6 --batch-size 8 --gradient-accumulation 2 \
  --num-generations 8 --max-completion-length 512 --beta 0.005 \
  --save-steps 50 --seed 20260917
```

SFT 어댑터 4종은 전부 이 스크립트 하나로 학습하며, 어댑터별 정확한 인자(LR, batch, 시드)와
데이터 sha256 은 각 `checkpoints/*/training_metadata.json` 에 기록돼 있습니다.
verify voter 는 학습이 없습니다 (hybrid_3145 가중치 + 자기검증 프롬프트).

시드: SFT `2026`, GRPO `20260917`, 게이트용 N=8 풀 `20260924`, few-shot N=8 `20260925`.
GRPO 학습 로그는 `logs/grpo_scaleup_train.log` 에 전 구간이 남아 있습니다 (wandb 미사용).

### 제출본 재현 검증

리더보드 제출본이 이 문서에 적힌 규칙대로 만들어졌는지 확인하는 절차입니다.

```bash
bash scripts/reproduce_all.sh compose   # GPU 불필요, 1분이면 끝
```

생성 당시 샘플(`outputs/*.jsonl`)을 저장소에 보존해 뒀기 때문에, 답 선택 규칙(1~5단계)을
재실행해 제출본과 **바이트 단위(sha256)로 일치**하는지 검사할 수 있습니다.

`full` 모드는 샘플 생성부터 다시 돕니다. 다만 이 경로는 바이트 일치가 원리적으로 불가능합니다.
같은 시드라도 GPU 기종, 추론 엔진 버전, 요청 동시성(배치 구성)이 다르면 부동소수점 연산
순서가 달라져 경계 토큰에서 다른 샘플이 나오고(greedy 포함), TIR 단계는 코드 실행 타임아웃이
걸려 있어 실행 시점에도 좌우됩니다. 그래서 확률적인 생성과 결정적인 조립을 분리해, 조립
층만큼은 바이트 검증을 걸어 뒀습니다.

### 테스트 공개 이후 커밋 (2026-08-31, 재현성과 문서 한정)

최종 테스트 공개(8/31 00:00 KST) 이후의 커밋은 전부 **재현성 확보와 문서화** 목적이며,
추론 결과를 만드는 로직(프롬프트, 샘플링 파라미터, 후처리, 가중치)은 바꾸지 않았습니다.
답 생성, 추출, 선택 코드(`gen_*.py`, `extractor_v2.py`, `tir_common.py`,
`build_*_submission.py`)는 8/30 01:53 커밋 이후 변경이 없고, 제출에 쓰인 어댑터 5종은
그 이전에 학습된 것으로 이후 재학습하지 않았습니다.

| 커밋 | 변경 | 성격 |
|---|---|---|
| `06f3e4a` | 공식 train/test CSV 동봉, README 통합, `reproduce_all.sh compose` 를 6단계 재생성 + 고정 sha256 대조로 강화 | 검증 기록 강화, 문서 |
| `06f3e4a` | `mega_bands.py` 입출력 경로 인자화, `answer` 열 없는 입력(최종 테스트) 대응 | 실행 인자 |
| `06f3e4a` | `train_qlora.py` 길이 초과 샘플을 예외 대신 제외 | 학습 스크립트 (제출 어댑터와 무관) |
| `ebf5693` | `mega_holdout_run.sh` → `run_inference.sh` 개명, TIR 워커 수 환경변수화(기본값 64 유지) | 문서, 실행 인자 |
| `b494f14` | self-consistency 재개 가드의 행수 하드코딩(2000)을 입력 행수 기준으로 수정 | 필터링된 테스트셋 실행 대응 |

성격상 오해의 소지가 있는 두 가지는 따로 밝혀 둡니다.

1. **TIR 생성 모델 기본값을 `tirsft` → `hybrid3145` 로 수정**(`run_inference.sh`).
   리더보드 제출본의 TIR 풀은 전부 `hybrid3145` 로 만든 것이고
   `reproduce_all.sh compose` 가 이를 sha256 바이트 일치로 증명합니다. `tirsft` 는 채택되지
   않은 실험 어댑터로, 리허설 스크립트에만 남아 있던 값이었습니다. 제출물에 맞춰 코드를 고친 것이지 결과를 바꾼 수정이 아닙니다.
2. **few-shot 포인터 게이트 재료 생성 단계 추가**(`run_inference.sh` 의 `fs3_greedy`,
   `fs3_n8`, `h3145_n8lp`, `ck150_n8lp`). 게이트 규칙 자체는 8/30 제출본에서 확정된 것이며,
   실행 스크립트에 빠져 있던 생성 단계를 채워 코드만으로 체인을 재현할 수 있게 한 것입니다.

최종 추론 때 쓴 동시성 값은 `TIR_REQ_WORKERS=256`, `TIR_EXEC_WORKERS=192` 입니다
(코드 기본값은 64 그대로). 동시성은 샘플링 파라미터가 아니지만 vLLM 의 배치 구성을 바꾸므로
재현 조건으로 함께 적어 둡니다.

## 체크포인트

| 어댑터 | 학습 데이터 | LR | seed |
|---|---|---|---|
| hybrid_3145_r8_qv_lr2e6_e1 | hybrid_3145.jsonl (3,145) | 2e-6 | 2026 |
| hybrid_3244_r8_qv_lr2e6_e1 | hybrid_3244.jsonl (3,244) | 2e-6 | 2026 |
| external_3000_r8_qv_lr2e6_e1 | external_math_3000.jsonl (3,000) | 2e-6 | 2026 |
| hybrid_4145_r8_qv_lr1p5e6_e1 | hybrid_4145.jsonl (4,145) | 1.5e-6 | 2026 |
| grpo_.../checkpoint-150 | grpo_passrate_scaleup.jsonl (667) | 2e-6 | 20260917 |

전부 Qwen2.5-3B-Instruct 위의 LoRA(r8, q/v)입니다. 하이퍼파라미터와 데이터 sha256은
각 디렉터리의 `training_metadata.json` 에 있고, verify voter 는 별도 가중치가 아니라
hybrid_3145 에 자기검증 프롬프트를 얹은 것입니다.

## 사용 데이터

파일별 구성, sha256, 오염 제거 기록은 `data/**/*.manifest.json` 에 기계 판독 가능한
형태로 남겨 뒀습니다.

**대회 공식 데이터**

| 파일 | 용도 |
|---|---|
| `deep_chal_math_dataset_train.csv` (17,000) | SFT 소재 선별, GRPO 풀, 홀드아웃 검증셋 |
| `deep_chal_math_dataset_test.csv` (2,000) | 최종 추론 대상, **학습에 사용하지 않음** |
| `deep_chal_math_leaderboard_filtered.csv` | 제출본 재현 검증(compose)이 참조하는 문제 목록 |

**외부 공개 데이터**: 전부 무료 공개이고, 선별은 sha256 기반 고정 시드 샘플링으로
결정적입니다(`scripts/prepare_hard_math_sft.py`). 공식 train과 평가 문제와의 중복은
정규화 exact match + token-Jaccard / SequenceMatcher 근사 검사로 걸렀습니다
(manifest 의 `official_decontamination`).

| 데이터셋 | 리비전 | 라이선스 | 사용처 |
|---|---|---|---|
| [AI-MO/NuminaMath-1.5](https://huggingface.co/datasets/AI-MO/NuminaMath-1.5) | `1b05109f` | Apache-2.0 | external_3000 일부, hybrid_3145 내 300, hybrid_4145 추가분 1,000 |
| [openai/gsm8k](https://huggingface.co/datasets/openai/gsm8k) | `740312a` | MIT | external_3000 (1,000) |
| [DigitalLearningGmbH/MATH-lighteval](https://huggingface.co/datasets/DigitalLearningGmbH/MATH-lighteval) | `92ace7ed` | MIT | external_3000 (1,200), hybrid_3145 내 700 |

**상용 API 생성 CoT**: `hybrid_3145.jsonl` 풀이 중 2,145개는 상용 API로 생성했습니다.
학습 데이터 구축에만 썼고, test 문제를 외부 API에 넣거나 추론 시점에 외부 모델을
호출한 일은 없습니다. 나머지 풀이는 사람이 쓴 공개 풀이(MATH-lighteval 700,
NuminaMath 큐레이션 300)이고, 문항 원문은 공식 train과 위 공개 데이터셋에서만 왔습니다.

| teacher | 샘플 수 |
|---|---|
| gpt-5.6-luna | 1,906 |
| gpt-5.4-mini-2026-03-17 | 133 |
| gpt-5.6-luna-high | 82 |
| gpt-5.6-terra | 19 |
| gpt-5.4-mini-high | 5 |

**GRPO 풀**: 공식 train에서 hybrid_3145 의 stochastic pass-rate 가 2~6/8인 667문제를
선별했습니다(`grpo_passrate_scaleup.jsonl`). 보상은 공식 정답 라벨과의 exact match 뿐이고,
다른 모델의 채점이나 판정은 개입하지 않습니다.

**기타**: TIR(Tool-Integrated Reasoning, 모델이 쓴 코드를 실행해 답을 검증)은
데이터가 아니라 추론 기법입니다. 실행은 전부 로컬 파이썬
서브프로세스이고 sympy(BSD) 등 표준 공개 라이브러리만 씁니다.

## 환경

요건(Python ≥ 3.10, PyTorch ≥ 2.0, CUDA ≥ 12.0)은 두 환경 모두 충족하며, 버전은
재현성을 위해 전부 `==` 로 고정했습니다. 설치와 용도 구분은 [사전 준비](#사전-준비) 참고.

| 환경 | Python | PyTorch | CUDA |
|---|---|---|---|
| `requirements.txt` (시스템 python3) | 3.12.3 | 2.8.0+cu128 | 12.8 |
| `requirements-vllm.txt` (venv) | 3.12.3 | 2.13.0 | 13.0 |

최종 추론 환경: RTX PRO 6000 Blackwell 96GB, driver 595.91.07
(동작 확인: RTX 5090 32GB, A100 80GB, Blackwell 96GB).

## 저장소 구성

```
checkpoints/   LoRA 어댑터 5종 + training_metadata.json  (최종 모델 체크포인트)
data/          공식 train/test CSV + 학습 데이터 + manifest
outputs/       제출본 재현 검증용 추론 산출물 (voter/SC/TIR/게이트 재료)
scripts/       학습, 추론, 조립 스크립트
logs/          GRPO 학습 로그
REPORT.md      실험 보고서 (Experiment Report). 접근 방법, 전략, 검증 방법론
```

### 스크립트 지도

직접 실행하는 진입점은 6개이고, 나머지는 전부 이들이 내부에서 호출하거나 import 합니다.

| 순서 | 진입점 | 하는 일 | 내부에서 쓰는 것 |
|---|---|---|---|
| 0 | `bootstrap_pod.sh` | 새 pod 의존성 복구 | 없음 |
| 1 | `prepare_hard_math_sft.py` → `merge_sft_jsonl.py` | SFT 데이터 선별과 병합 | 없음 |
| 2 | `train_qlora.py` | LoRA SFT (어댑터 4종) | 없음 |
| 3 | `screen_grpo_passrate.py` → `build_grpo_passrate_pool.py` | GRPO 풀 선별 | submit_baseline |
| 4 | `train_grpo_qlora.py` | GRPO 학습 (ck150) | 없음 |
| 5 | `vllm_server.sh` + `run_inference.sh` | 추론 재료 전 층 생성 | gen_client, gen_fewshot_client, gen_n8_logprobs, tir_repair_client, mega_bands, screen_grpo_passrate |
| 6 | `compose_final_submissions.py` | 재료 → 제출 CSV 조립 (+answer 있으면 채점) | extractor_v2, tir_common |
| 검증 | `reproduce_all.sh` | 제출본 조립 재현 검증 (compose/full) | baseline, rebuild_chain, build_* 5종, build_sc_source, extend_self_consistency_samples |

공용 모듈: `extractor_v2.py`(답 추출), `tir_common.py`(코드 실행과 정규화),
`submit_baseline.py`(프롬프트 정의), `ensemble_predictions.py`(정수 정규화)
