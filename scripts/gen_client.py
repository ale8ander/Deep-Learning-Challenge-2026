"""상주 vLLM 서버로 일반(비-TIR) 생성을 돌린다 — baseline.py 의 서버판.

baseline.py 와 같은 규약을 쓴다: 같은 system 프롬프트, greedy, 절단 시 더 긴 재시도.
다른 점은 답 추출기가 **v2**(Final answer 최우선 + LaTeX 래퍼 허용)라는 것뿐이다.
원문 `response` 를 그대로 저장하므로 나중에 다른 추출기로 재파싱할 수 있다.

용도: holdout464 voter 예측 생성(챔피언을 464에서 직접 채점하기 위해 필요),
그리고 임의의 프롬프트 스타일 실험.
"""
import argparse
import csv
import json
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402
from submit_baseline import SYSTEM_PROMPTS  # noqa: E402


def read_rows(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in open(p) if line.strip()]
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="hybrid3145")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--prompt-style", default="default", choices=sorted(SYSTEM_PROMPTS))
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--retry-max-new-tokens", type=int, default=2048)
    ap.add_argument("--request-workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)

    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]
    system = SYSTEM_PROMPTS[args.prompt_style]

    def gen(question, max_tokens):
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": question}],
            max_tokens=max_tokens, temperature=0.0, seed=args.seed)
        c = resp.choices[0]
        return (c.message.content or ""), c.finish_reason

    started = time.time()
    with ThreadPoolExecutor(max_workers=args.request_workers) as pool:
        first = list(pool.map(lambda r: gen(r["question"], args.max_new_tokens), rows))

    # 절단된 것만 더 긴 예산으로 재생성한다(baseline.py 의 retry 규약).
    retry_idx = [i for i, (_, fr) in enumerate(first) if fr == "length"]
    if retry_idx:
        with ThreadPoolExecutor(max_workers=args.request_workers) as pool:
            redone = list(pool.map(lambda i: gen(rows[i]["question"], args.retry_max_new_tokens), retry_idx))
        for i, res in zip(retry_idx, redone):
            first[i] = res

    correct = total = 0
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for row, (text, finish) in zip(rows, first):
            pred = norm(extract_v2(text))
            gold = norm(row.get("answer"))
            rec = {"id": row["id"], "prediction": None if pred is None else str(pred),
                   "response": text, "retried_truncated": finish == "length"}
            if gold is not None:
                rec["answer"] = str(gold)
                rec["correct"] = pred is not None and pred == gold
                total += 1
                correct += int(rec["correct"])
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    elapsed = time.time() - started
    score = f"{correct}/{total}" if total else "(정답 없음)"
    print(f"{args.model}/{args.prompt_style}: {score}  retry {len(retry_idx)}  "
          f"{elapsed:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
