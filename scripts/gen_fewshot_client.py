"""few-shot greedy 클라이언트 — gen_client.py 규약(v2 추출기, 1024→2048 retry)에
예시 user/assistant 턴을 앞에 붙인 버전. 이 프로젝트에서 few-shot 은 첫 시도다.
"""
import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402
from submit_baseline import SYSTEM_PROMPTS  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--exemplars", default="outputs/fewshot_exemplars3.json")
    ap.add_argument("--model", default="hybrid3145")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--retry-max-new-tokens", type=int, default=2048)
    ap.add_argument("--request-workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--num-samples", type=int, default=1,
                    help=">1 이면 temp 0.7 stochastic N샘플, predictions 리스트 저장")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)
    shots = json.load(open(ROOT / args.exemplars))
    system = SYSTEM_PROMPTS["default"]
    prefix = [{"role": "system", "content": system}]
    for s in shots:
        prefix += [{"role": "user", "content": s["user"]},
                   {"role": "assistant", "content": s["assistant"]}]

    p = Path(args.input)
    rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    if args.limit:
        rows = rows[: args.limit]

    lock = threading.Lock()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = args.output.open("w", encoding="utf-8")
    stats = {"done": 0, "correct": 0}
    t0 = time.time()

    def gen(row, max_tokens):
        n = args.num_samples
        r = client.chat.completions.create(
            model=args.model,
            messages=prefix + [{"role": "user", "content": row["question"]}],
            max_tokens=max_tokens,
            temperature=0.0 if n == 1 else 0.7,
            top_p=1.0 if n == 1 else 0.95,
            n=n, seed=args.seed)
        c = r.choices[0]
        return r, (c.message.content or ""), c.finish_reason == "length"

    def worker(row):
        resp, text, trunc = gen(row, args.max_new_tokens)
        if trunc and args.num_samples == 1:
            resp, text, trunc = gen(row, args.retry_max_new_tokens)
        pred = norm(extract_v2(text))
        gold = norm(str(row.get("answer", "")).replace(",", ""))
        rec = {"id": row["id"], "answer": None if gold is None else str(gold),
               "prediction": None if pred is None else str(pred),
               "correct": pred is not None and gold is not None and str(pred) == str(gold),
               "retried_truncated": trunc, "response": text}
        if args.num_samples > 1:
            preds = [norm(extract_v2(c.message.content or "")) for c in resp.choices]
            rec["predictions"] = [None if p is None else str(p) for p in preds]
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            stats["done"] += 1
            stats["correct"] += int(rec["correct"])
            if stats["done"] % 100 == 0:
                print(f"[{stats['done']}/{len(rows)}] 정답 {stats['correct']} "
                      f"{time.time()-t0:.0f}s", flush=True)

    with ThreadPoolExecutor(max_workers=args.request_workers) as ex:
        list(ex.map(worker, rows))
    out.close()
    print(f"완료: {stats['done']}건, 정답 {stats['correct']}, {time.time()-t0:.0f}초")


if __name__ == "__main__":
    main()
