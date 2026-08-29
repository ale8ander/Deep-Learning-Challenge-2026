"""상주 vLLM 서버에 붙는 TIR 하네스 — 매 실험 8분 기동 비용을 없앤다.

`scripts/tir_inference_vllm.py`와 동작·출력 포맷이 동일하고, 모델을 프로세스 안에서
올리는 대신 `scripts/vllm_server.sh`가 띄운 서버에 HTTP로 요청한다.
어댑터는 서버에 등록된 이름으로 고른다(--model hybrid3145 / tirsft / grpo96 / verbose).

프롬프트 상수는 tir_inference.py에서 직접 import한다(복붙 금지 — 학습/추론 불일치 방지).

⚠️ 추론 시점 코드 실행은 대회 규정 회색지대다. CONTEXT "TIR" 절 참고.
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tir_common import (  # noqa: E402
    TIR_SYSTEM, FINAL_NUDGE, CODE_BLOCK, extract_answer, normalize, run_code,
)


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
    ap.add_argument("--model", default="hybrid3145", help="서버에 등록된 어댑터 이름")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--final-max-new-tokens", type=int, default=1024)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--exec-timeout", type=int, default=60)
    ap.add_argument("--exec-workers", type=int, default=96)
    ap.add_argument("--request-workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)

    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]
    N = max(1, args.num_samples)

    def chat(messages, max_tokens, n):
        """서버에 요청하고 completion 텍스트 n개를 순서대로 돌려준다."""
        kwargs = dict(model=args.model, messages=messages, max_tokens=max_tokens, n=n, seed=args.seed)
        if n > 1:
            kwargs.update(temperature=args.temperature, top_p=args.top_p)
        else:
            kwargs.update(temperature=0.0)
        resp = client.chat.completions.create(**kwargs)
        texts = [c.message.content or "" for c in sorted(resp.choices, key=lambda c: c.index)]
        return texts + [""] * (n - len(texts))

    started = time.time()
    base_convos = [
        [{"role": "system", "content": TIR_SYSTEM}, {"role": "user", "content": r["question"]}]
        for r in rows
    ]

    # 1라운드: 문제당 n개 샘플. 문제 단위로 병렬 요청한다.
    with ThreadPoolExecutor(max_workers=args.request_workers) as pool:
        first_per_row = list(pool.map(lambda c: chat(c, args.max_new_tokens, N), base_convos))
    first = [t for texts in first_per_row for t in texts]  # 인덱스: r*N + k

    # 코드 실행 (CPU 병렬)
    blocks = [CODE_BLOCK.findall(t) for t in first]
    with ThreadPoolExecutor(max_workers=args.exec_workers) as pool:
        futures = [pool.submit(run_code, b[-1], args.exec_timeout) if b else None for b in blocks]
        executions = [f.result() if f is not None else None for f in futures]

    # 2라운드
    second_convos = []
    for idx, (text, execution) in enumerate(zip(first, executions)):
        if execution is None:
            feedback = ("No Python code block was found in your response. "
                        "Solve the problem directly and end with exactly: Final answer: <integer>")
        else:
            body = execution["stdout"].strip() or "(no output)"
            if execution["status"] != "ok":
                body += f"\n[{execution['status']}] {execution['stderr'].strip()[:400]}"
            feedback = f"Program output:\n```\n{body}\n```\n{FINAL_NUDGE}"
        second_convos.append(base_convos[idx // N] + [
            {"role": "assistant", "content": text},
            {"role": "user", "content": feedback},
        ])
    with ThreadPoolExecutor(max_workers=args.request_workers) as pool:
        second = [t[0] for t in pool.map(lambda c: chat(c, args.final_max_new_tokens, 1), second_convos)]

    def verified(execution, prediction):
        if execution is None or execution["status"] != "ok" or prediction is None:
            return False
        out = normalize((execution.get("stdout") or "").strip())
        return out is not None and out == prediction

    stats = {"code_found": 0, "exec_ok": 0, "exec_error": 0, "exec_timeout": 0,
             "correct": 0, "total": 0, "adopted": 0}
    args.output.parent.mkdir(parents=True, exist_ok=True)
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
            base_idx = r * N
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
                "code_executed": executions[base_idx] is not None,
                "exec_status": sample_status[0],
                "exec_stdout": (executions[base_idx]["stdout"][:500] if executions[base_idx] else None),
                "round1": first[base_idx], "round2": second[base_idx],
            }, ensure_ascii=False) + "\n")

    stats["elapsed_sec"] = round(time.time() - started, 1)
    print(json.dumps(stats, ensure_ascii=False))
    print(f"correct={stats['correct']}/{stats['total']} elapsed={stats['elapsed_sec']:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
