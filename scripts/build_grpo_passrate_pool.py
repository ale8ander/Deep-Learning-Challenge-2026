"""스크리닝 결과에서 pass-rate 25~75% 밴드를 골라 GRPO 학습 데이터로 변환.

grpo_passrate_94 와 같은 기준: N=8 중 정답 2~6개. 포맷도 동일
(id / prompt(chat) / answer). hybrid_3145 SFT 학습 문제와의 중복은 제외한다
(기존 94 풀의 sft_hybrid3145_overlap=0 관행 유지).
"""
import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

SYSTEM_PROMPT = (
    "Solve the math problem carefully. The answer is always an integer. "
    "End your response with exactly: Final answer: <integer>"
)


def read_jsonl(path):
    return [json.loads(l) for l in open(path) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--screen", type=Path, required=True)
    ap.add_argument("--pool-csv", type=Path, required=True)
    ap.add_argument("--sft-jsonl", type=Path,
                    default=Path("data/processed/hybrid_3145.jsonl"))
    ap.add_argument("--min-correct", type=int, default=2)
    ap.add_argument("--max-correct", type=int, default=6)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    questions = {r["id"].strip(): r for r in csv.DictReader(
        open(args.pool_csv, encoding="utf-8-sig"))}
    sft_ids = {str(r.get("id", "")).strip() for r in read_jsonl(args.sft_jsonl)}
    screen = read_jsonl(args.screen)

    hist = Counter(r["n_correct"] for r in screen)
    selected, sft_skipped = [], 0
    for r in screen:
        if not args.min_correct <= r["n_correct"] <= args.max_correct:
            continue
        if r["id"] in sft_ids:
            sft_skipped += 1
            continue
        q = questions[r["id"]]
        selected.append({
            "id": r["id"],
            "prompt": [{"role": "system", "content": SYSTEM_PROMPT},
                       {"role": "user", "content": q["question"]}],
            "answer": str(q["answer"]).strip().replace(",", ""),
            "screen_n_correct": r["n_correct"],
        })

    selected.sort(key=lambda x: x["id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for rec in selected:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    manifest = {
        "output": str(args.output),
        "output_sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
        "screen_input": str(args.screen),
        "screened": len(screen),
        "samples": len(selected),
        "band": f"n_correct {args.min_correct}..{args.max_correct} / 8",
        "band_histogram": {str(k): v for k, v in sorted(hist.items())},
        "sft_hybrid3145_skipped": sft_skipped,
    }
    Path(str(args.output).replace(".jsonl", ".manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
