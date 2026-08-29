"""5-voter 산출물 -> self-consistency 생성용 소스 jsonl.

왜 필요한가: `extend_self_consistency_samples.py` 는 `--min/--max-baseline-support` 로
support tier 를 거르는데, 그 값을 **소스 jsonl 의 `baseline_support` 필드에서** 읽는다.
voter 하나의 산출물에는 그 필드가 없다(있을 수가 없다 — support 는 5개 voter 의 합의도다).
그대로 넘기면 `.get("baseline_support", 0)` 이 0 을 돌려줘 **전 행이 조용히 탈락**하고
빈 파일이 나온다. 에러도 안 난다.

이 스크립트가 그 빠진 고리다: 5-voter 다수결과 support 를 계산해 SC 생성이 요구하는
필드(id/question/answer/votes/baseline_prediction/baseline_support)를 갖춘 행을 만든다.

동률 규약은 `rebuild_chain.build()` 의 5-voter 단계와 동일하다:
최다 득표가 유일하고 2표 이상이면 채택, 아니면 voter1(hybrid_3145) 의 답.
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v1, norm  # noqa: E402


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--voters", nargs="+", required=True,
                    help="voter jsonl 5개. **첫 번째가 동률 fallback**(voter1=hybrid_3145)")
    ap.add_argument("--questions", type=Path,
                    default=ROOT / "data/deep_chal_math_leaderboard_filtered.csv")
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    src = list(csv.DictReader(open(args.questions, encoding="utf-8-sig")))
    order = [r["id"] for r in src]
    qmap = {r["id"]: r for r in src}

    preds = []
    for path in args.voters:
        m = {r["id"]: norm(extract_v1(r.get("response"))) for r in read_jsonl(path)}
        preds.append(m)
    if len(preds) < 2:
        raise SystemExit("voter 가 2개 미만이면 support 가 의미 없다")

    n_written = 0
    dist = Counter()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for pid in order:
            votes = [p.get(pid) for p in preds]
            counts = Counter(v for v in votes if v is not None)
            support = counts.most_common(1)[0][1] if counts else 0
            if counts:
                top = max(counts.values())
                winners = [a for a, c in counts.items() if c == top]
                pick = winners[0] if (len(winners) == 1 and top >= 2) else votes[0]
            else:
                pick = votes[0]
            row = qmap[pid]
            f.write(json.dumps({
                "id": pid,
                "question": row["question"],
                "answer": norm(row.get("answer")),
                "votes": [None if v is None else str(v) for v in votes],
                "baseline_prediction": None if pick is None else str(pick),
                "baseline_support": support,
            }, ensure_ascii=False) + "\n")
            dist[support] += 1
            n_written += 1

    print(f"{n_written}행 -> {args.output}")
    print(f"support 분포: {dict(sorted(dist.items()))}")


if __name__ == "__main__":
    main()
