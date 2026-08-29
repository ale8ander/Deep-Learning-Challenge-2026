"""N=8 stochastic 생성 + 토큰 logprob 확신도 기록 (DeepConf 계열 실험용).

screen_grpo_passrate.py 와 같은 규약(단일 요청 n=8, temp 0.7)이되, 샘플마다
- mean_logprob: 전체 토큰 평균 logprob
- tail_logprob: 마지막 128토큰 평균 (답 직전 확신)
- min_group: 슬라이딩 윈도(128, stride 64) 평균 logprob 의 최솟값 (DeepConf 의
  lowest-group-confidence — 중간에 흔들린 트레이스를 잡는다)
를 저장한다. 예측 추출은 extractor_v2 (배포와 동일).
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


def read_rows(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(l) for l in open(p) if l.strip()]
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def conf_stats(logprobs, win=128, stride=64):
    if not logprobs:
        return None, None, None
    mean = sum(logprobs) / len(logprobs)
    tail = logprobs[-win:]
    tail_mean = sum(tail) / len(tail)
    groups = []
    for s in range(0, max(1, len(logprobs) - win + 1), stride):
        g = logprobs[s:s + win]
        groups.append(sum(g) / len(g))
    return mean, tail_mean, min(groups)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="hybrid3145")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--request-workers", type=int, default=24)
    ap.add_argument("--seed", type=int, default=20260924)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)

    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]
    done = set()
    if args.resume and args.output.exists():
        done = {json.loads(l)["id"] for l in open(args.output) if l.strip()}
        print(f"resume: {len(done)}개 건너뜀", flush=True)
    todo = [r for r in rows if r["id"] not in done]

    system = SYSTEM_PROMPTS["default"]
    lock = threading.Lock()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = args.output.open("a", encoding="utf-8")
    stats = {"done": 0, "errors": 0}
    t0 = time.time()

    def run_one(row):
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": row["question"]}],
            max_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            n=args.num_samples, seed=args.seed, logprobs=True)
        gold = norm(str(row.get("answer", "")).replace(",", ""))
        preds, confs, truncated = [], [], 0
        for c in resp.choices:
            pred = norm(extract_v2(c.message.content or ""))
            preds.append(None if pred is None else str(pred))
            lps = [t.logprob for t in (c.logprobs.content or [])] if c.logprobs else []
            m, tl, mg = conf_stats(lps)
            confs.append({"mean": m, "tail": tl, "min_group": mg, "ntok": len(lps)})
            truncated += int(c.finish_reason == "length")
        n_correct = sum(1 for p in preds if gold is not None and p == str(gold))
        return {"id": row["id"], "answer": None if gold is None else str(gold),
                "num_samples": args.num_samples, "n_correct": n_correct,
                "truncated": truncated, "predictions": preds, "confidence": confs}

    def worker(row):
        try:
            rec = run_one(row)
        except Exception as e:  # noqa: BLE001
            with lock:
                stats["errors"] += 1
                print(f"[err] {row['id']}: {e}", flush=True)
            return
        with lock:
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out.flush()
            stats["done"] += 1
            if stats["done"] % 50 == 0:
                el = time.time() - t0
                print(f"[{stats['done']}/{len(todo)}] {el:.0f}s "
                      f"{stats['done']/el:.2f}문제/s", flush=True)

    with ThreadPoolExecutor(max_workers=args.request_workers) as ex:
        list(ex.map(worker, todo))
    out.close()
    print(f"완료: {stats['done']}건, 오류 {stats['errors']}건, "
          f"{time.time()-t0:.0f}초", flush=True)


if __name__ == "__main__":
    main()
