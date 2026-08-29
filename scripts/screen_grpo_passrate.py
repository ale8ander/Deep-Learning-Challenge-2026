"""GRPO 스케일업용 pass-rate 스크리닝 — 상주 vLLM 서버 경유 N샘플 롤아웃.

gen_client.py 의 stochastic 판. 문제당 한 요청에 n=N 을 실어 8샘플을 받고,
extractor_v2 로 정답 여부만 채점해 pass rate 를 기록한다. 전체 응답 본문은
저장하지 않는다(밴드 선별에는 예측값만 필요).

무인 장기 실행을 전제로 하므로 문제 단위로 완료 즉시 append 하고,
--resume 이면 출력 파일에 이미 있는 id 는 건너뛴다.

실행은 /workspace/venv-vllm/bin/python (openai 패키지) 로.
"""
import argparse
import csv
import json
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
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
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--request-workers", type=int, default=48)
    ap.add_argument("--seed", type=int, default=20260916)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--save-responses", action="store_true")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)

    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]

    done_ids = set()
    if args.resume and args.output.exists():
        for line in open(args.output):
            if line.strip():
                done_ids.add(json.loads(line)["id"])
        print(f"resume: {len(done_ids)}개 완료분 건너뜀", flush=True)
    todo = [r for r in rows if r["id"] not in done_ids]

    system = SYSTEM_PROMPTS["default"]
    lock = threading.Lock()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    out = args.output.open("a", encoding="utf-8")
    stats = {"done": 0, "errors": 0}
    started = time.time()

    def run_one(row):
        resp = client.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": system},
                      {"role": "user", "content": row["question"]}],
            max_tokens=args.max_new_tokens,
            temperature=args.temperature, top_p=args.top_p,
            n=args.num_samples, seed=args.seed)
        gold = norm(str(row.get("answer", "")).replace(",", ""))
        preds, truncated = [], 0
        for c in resp.choices:
            pred = norm(extract_v2(c.message.content or ""))
            preds.append(None if pred is None else str(pred))
            truncated += int(c.finish_reason == "length")
        n_correct = sum(1 for p in preds if gold is not None and p == str(gold))
        rec = {"id": row["id"], "answer": None if gold is None else str(gold),
                "num_samples": args.num_samples, "n_correct": n_correct,
                "pass_rate": n_correct / args.num_samples,
                "truncated": truncated, "predictions": preds}
        if args.save_responses:
            rec["responses"] = [c.message.content or "" for c in resp.choices]
        return rec

    def worker(row):
        try:
            rec = run_one(row)
        except Exception as exc:  # noqa: BLE001 — 개별 실패는 기록만 하고 계속
            rec = None
            with lock:
                stats["errors"] += 1
                print(f"[err] {row['id']}: {exc}", flush=True)
        if rec is not None:
            with lock:
                out.write(json.dumps(rec, ensure_ascii=False) + "\n")
                out.flush()
                stats["done"] += 1
                if stats["done"] % 100 == 0:
                    el = time.time() - started
                    rate = stats["done"] / el
                    eta = (len(todo) - stats["done"]) / rate if rate else 0
                    print(f"[{stats['done']}/{len(todo)}] {el:.0f}s 경과, "
                          f"{rate:.2f}문제/s, ETA {eta/60:.0f}분", flush=True)

    with ThreadPoolExecutor(max_workers=args.request_workers) as pool:
        futures = [pool.submit(worker, r) for r in todo]
        for f in as_completed(futures):
            f.result()

    out.close()
    print(f"완료: {stats['done']}건, 오류 {stats['errors']}건, "
          f"{time.time() - started:.0f}초", flush=True)


if __name__ == "__main__":
    main()
