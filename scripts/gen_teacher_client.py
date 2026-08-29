"""teacher 모델(32B)로 문제 풀이를 N개씩 생성한다 — 학습 데이터 구축 전용.

`gen_client.py` 는 greedy 단발이라 N샘플·temperature 가 없어서 별도로 둔다.
프롬프트는 **우리 추론 스택과 같은 포맷**(`Final answer: <정수>`)을 강제한다.
teacher 가 다른 포맷으로 답하면 그 데이터로 학습한 3B 도 다른 포맷으로 답하게 되고,
그러면 추출기·5-voter·SC·TIR 전 스택이 깨진다.
"""
import argparse
import json
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from openai import OpenAI
from tqdm import tqdm

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))

SYSTEM = ("Solve the math problem. Think step by step, showing your reasoning "
          "concisely. End your response with exactly:\nFinal answer: <integer>")


def read_pool(path):
    p = Path(path)
    rows = []
    if p.suffix == ".jsonl":
        for line in open(p, encoding="utf-8"):
            if line.strip():
                rows.append(json.loads(line))
    else:
        import csv
        rows = list(csv.DictReader(open(p, encoding="utf-8-sig")))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--base-url", default="http://localhost:8001/v1")
    ap.add_argument("--model", default="teacher32b")
    ap.add_argument("--num-samples", type=int, default=4)
    ap.add_argument("--max-new-tokens", type=int, default=1536)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--request-workers", type=int, default=64)
    ap.add_argument("--seed", type=int, default=20260930)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    rows = read_pool(args.input)
    if args.limit:
        rows = rows[:args.limit]
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800, max_retries=3)

    def one(row):
        try:
            r = client.chat.completions.create(
                model=args.model,
                messages=[{"role": "system", "content": SYSTEM},
                          {"role": "user", "content": row["question"]}],
                n=args.num_samples, temperature=args.temperature, top_p=0.95,
                max_tokens=args.max_new_tokens, seed=args.seed,
            )
            return {"id": row["id"], "question": row["question"],
                    "answer": row.get("answer"),
                    "responses": [c.message.content or "" for c in r.choices]}
        except Exception as e:                      # 한 문제 실패로 전체를 죽이지 않는다
            return {"id": row["id"], "question": row["question"],
                    "answer": row.get("answer"), "responses": [], "error": str(e)[:200]}

    args.output.parent.mkdir(parents=True, exist_ok=True)
    tmp = Path(str(args.output) + ".part")
    ok = fail = 0
    with tmp.open("w", encoding="utf-8") as f, \
            ThreadPoolExecutor(max_workers=args.request_workers) as ex:
        for res in tqdm(ex.map(one, rows), total=len(rows), desc="teacher"):
            f.write(json.dumps(res, ensure_ascii=False) + "\n")
            ok += bool(res.get("responses"))
            fail += not res.get("responses")
    tmp.replace(args.output)
    print(json.dumps({"total": len(rows), "ok": ok, "failed": fail,
                      "output": str(args.output)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
