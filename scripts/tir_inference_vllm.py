"""TIR 하네스의 vLLM 포팅 — `scripts/tir_inference.py`와 동일한 규약, 훨씬 빠른 실행.

HF `generate()`는 배치 안 모든 시퀀스가 끝날 때까지 기다리는 정적 배칭이라
어려운 문제(=길게 생성)가 섞이면 GPU가 놀았다. vLLM은 continuous batching이라
끝난 시퀀스 자리에 새 요청을 바로 채운다. 측정상 HF 배치당 357초짜리 작업 기준.

⚠️ 두 가지 주의
  1) 커널이 달라 같은 greedy여도 HF와 출력이 미세하게 다를 수 있다.
     `--compare-with` 로 기존 HF 결과와 예측 일치율을 반드시 확인하고 쓸 것.
  2) 추론 시점 코드 실행은 대회 규정 회색지대다. CONTEXT "TIR 스모크" 절 참고.

실행: /workspace/venv-vllm/bin/python scripts/tir_inference_vllm.py ...
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

DEFAULT_MODEL_PATH = "/workspace/models/Qwen2.5-3B-Instruct"

TIR_SYSTEM = (
    "You are a meticulous contest mathematician with a Python interpreter available.\n"
    "Solve the problem by writing Python code that computes the answer.\n"
    "Prefer brute-force enumeration, exhaustive case checking, and sympy over hand algebra — "
    "the interpreter is exact and fast, so verify rather than guess.\n"
    "Write exactly one Python code block in this format:\n"
    "```python\n"
    "# your code; it MUST print the result\n"
    "print(result)\n"
    "```\n"
    "Then stop and wait. You will be shown the program output, and only then give the final answer.\n"
    "The answer is always an integer. End your final response with exactly: Final answer: <integer>"
)

FINAL_NUDGE = (
    "Program output is above. If it answers the problem, state the answer now. "
    "If the output is wrong, empty, or an error, reason it out yourself instead. "
    "End with exactly: Final answer: <integer>"
)

CODE_BLOCK = re.compile(r"```(?:python|py)?\s*\n(.*?)```", re.S)
ANSWER_PATTERNS = (
    re.compile(r"(?:final answer|정답)\s*(?:is|:|=)?\s*[^\d\-]{0,15}?(-?\d[\d,]*)", re.I),
    re.compile(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}"),
    re.compile(r"(-?\d[\d,]*)"),
)


def extract_answer(text):
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].replace(",", "")
    return None


def normalize(value):
    """정수로 정규화. inf/nan은 None으로 버린다.

    모델이 생성한 프로그램이 inf를 출력하는 경우가 실제로 있었다(N=8 샘플링에서 발생).
    float('inf')는 파싱에 성공하고 int()에서 OverflowError가 나므로 반드시 함께 잡아야 한다.
    """
    if value is None:
        return None
    s = str(value).strip()
    if not s:
        return None
    try:
        return int(s)
    except ValueError:
        pass
    try:
        f = float(s)
    except (ValueError, OverflowError):
        return None
    if f != f or f in (float("inf"), float("-inf")):
        return None
    try:
        return int(f)
    except (OverflowError, ValueError):
        return None


def run_code(code, timeout):
    with tempfile.TemporaryDirectory() as workdir:
        script = Path(workdir) / "solve.py"
        script.write_text(code, encoding="utf-8")
        try:
            proc = subprocess.run(
                [sys.executable, "-I", str(script)],
                capture_output=True, text=True, timeout=timeout, cwd=workdir,
            )
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "stdout": "", "stderr": f"timed out after {timeout}s"}
        return {
            "status": "ok" if proc.returncode == 0 else "error",
            "stdout": proc.stdout[-2000:],
            "stderr": proc.stderr[-1000:],
        }


def read_rows(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in open(p)]
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--adapter-path", default=None)
    ap.add_argument("--model-path", default=DEFAULT_MODEL_PATH)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--final-max-new-tokens", type=int, default=1024)
    ap.add_argument("--exec-timeout", type=int, default=10)
    ap.add_argument("--exec-workers", type=int, default=32)
    ap.add_argument("--gpu-memory-utilization", type=float, default=0.85)
    ap.add_argument("--max-lora-rank", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-samples", type=int, default=1,
                    help=">1이면 TIR self-consistency. 샘플마다 코드를 따로 실행하고, "
                         "코드검증된 답들의 다수결을 최종 예측으로 쓴다.")
    ap.add_argument("--temperature", type=float, default=0.7, help="--num-samples>1 일 때만 사용")
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--min-count", type=int, default=2,
                    help="코드검증된 답 중 최다득표가 이 값 미만이면 예측을 내지 않는다(None).")
    ap.add_argument("--seed", type=int, default=20260827)
    ap.add_argument("--compare-with", type=Path, default=None,
                    help="기존 HF 결과 jsonl. 예측 일치율을 출력한다(엔진 차이 확인용).")
    args = ap.parse_args()

    from vllm import LLM, SamplingParams
    from vllm.lora.request import LoRARequest
    from transformers import AutoTokenizer

    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    llm = LLM(
        model=args.model_path,
        dtype="float16",
        enable_lora=args.adapter_path is not None,
        max_lora_rank=args.max_lora_rank,
        gpu_memory_utilization=args.gpu_memory_utilization,
        max_model_len=4096,
    )
    lora = LoRARequest("adapter", 1, args.adapter_path) if args.adapter_path else None

    def generate(prompts, max_tokens, n=1):
        """n>1이면 프롬프트당 n개 샘플을 평탄화해서 돌려준다(길이 = len(prompts)*n)."""
        if n > 1:
            params = SamplingParams(
                n=n, temperature=args.temperature, top_p=args.top_p,
                max_tokens=max_tokens, seed=args.seed,
            )
        else:
            params = SamplingParams(temperature=0.0, max_tokens=max_tokens)
        outs = llm.generate(prompts, params, lora_request=lora)
        # vLLM은 완료 순서대로 돌려줄 수 있으므로 request_id 기준으로 재정렬한다.
        outs = sorted(outs, key=lambda o: int(o.request_id))
        flat = []
        for o in outs:
            texts = [c.text for c in o.outputs]
            # n개를 요청했는데 덜 오면 빈 문자열로 채워 인덱스 정렬을 유지한다.
            texts += [""] * (n - len(texts))
            flat.extend(texts[:n])
        return flat

    started = time.time()
    N = max(1, args.num_samples)
    convos = [
        [{"role": "system", "content": TIR_SYSTEM}, {"role": "user", "content": r["question"]}]
        for r in rows
    ]
    prompts = [
        tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
        for c in convos
    ]
    # 인덱스 규약: 행 r의 k번째 샘플 = r*N + k
    first = generate(prompts, args.max_new_tokens, n=N)

    blocks = [CODE_BLOCK.findall(t) for t in first]
    with ThreadPoolExecutor(max_workers=args.exec_workers) as pool:
        futures = [
            pool.submit(run_code, b[-1], args.exec_timeout) if b else None
            for b in blocks
        ]
        executions = [f.result() if f is not None else None for f in futures]

    second_prompts = []
    for idx, (text, execution) in enumerate(zip(first, executions)):
        convo = convos[idx // N]
        if execution is None:
            feedback = ("No Python code block was found in your response. "
                        "Solve the problem directly and end with exactly: Final answer: <integer>")
        else:
            body = execution["stdout"].strip() or "(no output)"
            if execution["status"] != "ok":
                body += f"\n[{execution['status']}] {execution['stderr'].strip()[:400]}"
            feedback = f"Program output:\n```\n{body}\n```\n{FINAL_NUDGE}"
        new_convo = convo + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": feedback},
        ]
        second_prompts.append(
            tokenizer.apply_chat_template(new_convo, tokenize=False, add_generation_prompt=True)
        )
    second = generate(second_prompts, args.final_max_new_tokens, n=1)

    def verified(execution, prediction):
        """코드가 정상 실행돼 정수를 뱉었고, 모델이 그 값을 최종답으로 채택했는가."""
        if execution is None or execution["status"] != "ok" or prediction is None:
            return False
        stdout_value = normalize((execution.get("stdout") or "").strip())
        return stdout_value is not None and stdout_value == prediction

    stats = {"code_found": 0, "exec_ok": 0, "exec_error": 0, "exec_timeout": 0,
             "correct": 0, "total": 0, "adopted": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
    results = {}
    with args.output.open("w", encoding="utf-8") as out:
        for r, row in enumerate(rows):
            sample_preds, sample_status, verified_preds = [], [], []
            for k in range(N):
                idx = r * N + k
                execution = executions[idx]
                pred = normalize(extract_answer(second[idx])) or normalize(extract_answer(first[idx]))
                sample_preds.append(pred)
                sample_status.append(execution["status"] if execution else None)
                if execution is not None:
                    stats["code_found"] += 1
                    stats[f"exec_{execution['status']}"] += 1
                if verified(execution, pred):
                    verified_preds.append(pred)

            counts = Counter(verified_preds)
            top = counts.most_common()
            if top and top[0][1] >= args.min_count and not (len(top) > 1 and top[0][1] == top[1][1]):
                prediction, support = top[0]
            else:
                prediction, support = None, (top[0][1] if top else 0)

            answer = normalize(row.get("answer"))
            correct = prediction is not None and prediction == answer
            stats["total"] += 1
            stats["correct"] += int(correct)
            stats["adopted"] += int(prediction is not None)
            results[row["id"]] = prediction

            best_idx = r * N  # 대표 샘플(첫 번째)의 원문만 저장해 파일 크기를 억제한다
            out.write(json.dumps({
                "id": row["id"], "answer": answer, "prediction": prediction, "correct": correct,
                "verified_support": support,
                "verified_counts": {str(k2): v for k2, v in counts.most_common()},
                "sample_predictions": sample_preds,
                "sample_exec_status": sample_status,
                # 선택 연구용: 샘플별 코드와 실제 출력을 전부 남긴다.
                # (표만 세는 선택이 한계에 부딪혀, 코드 내용을 볼 수 있어야 한다)
                "sample_codes": [
                    (CODE_BLOCK.findall(first[r * N + k])[-1][:4000]
                     if CODE_BLOCK.findall(first[r * N + k]) else None)
                    for k in range(N)
                ],
                "sample_stdouts": [
                    (executions[r * N + k]["stdout"][:600] if executions[r * N + k] else None)
                    for k in range(N)
                ],
                "code_executed": executions[best_idx] is not None,
                "exec_status": sample_status[0],
                "exec_stdout": (executions[best_idx]["stdout"][:500] if executions[best_idx] else None),
                "round1": first[best_idx], "round2": second[best_idx],
            }, ensure_ascii=False) + "\n")

    stats["elapsed_sec"] = round(time.time() - started, 1)
    print(json.dumps(stats, ensure_ascii=False))
    print(f"correct={stats['correct']}/{stats['total']} elapsed={stats['elapsed_sec']:.0f}s -> {args.output}")

    if args.compare_with and args.compare_with.exists():
        hf = {}
        for line in open(args.compare_with):
            r = json.loads(line)
            hf[r["id"]] = normalize(r.get("prediction"))
        shared = [i for i in results if i in hf]
        same = sum(1 for i in shared if results[i] == hf[i])
        print(f"HF 결과와 비교: 공통 {len(shared)}문제 중 예측 일치 {same} ({same/max(len(shared),1):.1%})")


if __name__ == "__main__":
    main()
