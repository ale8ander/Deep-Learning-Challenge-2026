import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path


def load(path: Path) -> list[dict]:
    rows = []
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                rows.append(json.loads(line))
    return rows


def select_vote(predictions: list, baseline, min_count: int | None):
    counts = Counter(p for p in predictions if p is not None)
    if not counts:
        return baseline
    best = max(counts.values())
    winners = [value for value, count in counts.items() if count == best]
    if len(winners) != 1:
        return baseline
    if min_count is not None and best < min_count:
        return baseline
    return winners[0]


def select_best_logprob(predictions: list, logprobs: list, baseline):
    candidates = [
        (p, lp) for p, lp in zip(predictions, logprobs) if p is not None and lp is not None
    ]
    if not candidates:
        return baseline
    return max(candidates, key=lambda pair: pair[1])[0]


def select_confidence_weighted(predictions: list, logprobs: list, baseline, agg: str):
    grouped: dict = defaultdict(list)
    for p, lp in zip(predictions, logprobs):
        if p is not None and lp is not None:
            grouped[p].append(lp)
    if not grouped:
        return baseline
    if agg == "sum":
        scored = {value: sum(lps) for value, lps in grouped.items()}
    else:
        scored = {value: sum(lps) / len(lps) for value, lps in grouped.items()}
    return max(scored.items(), key=lambda pair: pair[1])[0]


def select_vote_then_logprob_tiebreak(predictions: list, logprobs: list, baseline, min_count: int):
    counts = Counter(p for p in predictions if p is not None)
    if not counts:
        return baseline
    best = max(counts.values())
    winners = [value for value, count in counts.items() if count == best]
    if len(winners) == 1 and best >= min_count:
        return winners[0]
    if len(winners) > 1:
        grouped: dict = defaultdict(list)
        for p, lp in zip(predictions, logprobs):
            if p in winners and lp is not None:
                grouped[p].append(lp)
        if grouped:
            avg = {value: sum(lps) / len(lps) for value, lps in grouped.items()}
            top = max(avg.items(), key=lambda pair: pair[1])[0]
            if best >= min_count:
                return top
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args()

    rows = load(args.input)
    total = len(rows)
    print(f"problems={total}")

    baseline_correct = sum(1 for r in rows if r["baseline_prediction"] == r["answer"])
    oracle = sum(
        1
        for r in rows
        if r["baseline_prediction"] == r["answer"] or r["answer"] in r["sample_predictions"]
    )
    print(f"baseline={baseline_correct}/{total} ({baseline_correct/total*100:.1f}%)")
    print(f"oracle(baseline OR any sample)={oracle}/{total} ({oracle/total*100:.1f}%)")
    print()

    strategies = {
        "pure_vote (no threshold)": lambda r: select_vote(
            r["sample_predictions"], r["baseline_prediction"], None
        ),
        "best_single_logprob": lambda r: select_best_logprob(
            r["sample_predictions"], r["sample_avg_logprob"], r["baseline_prediction"]
        ),
        "confidence_weighted_sum": lambda r: select_confidence_weighted(
            r["sample_predictions"], r["sample_avg_logprob"], r["baseline_prediction"], "sum"
        ),
        "confidence_weighted_avg": lambda r: select_confidence_weighted(
            r["sample_predictions"], r["sample_avg_logprob"], r["baseline_prediction"], "avg"
        ),
    }
    for min_count in range(2, min(9, len(rows[0]["sample_predictions"]) + 1)):
        strategies[f"min_count{min_count}"] = (
            lambda r, m=min_count: select_vote(r["sample_predictions"], r["baseline_prediction"], m)
        )
        strategies[f"min_count{min_count}+logprob_tiebreak"] = (
            lambda r, m=min_count: select_vote_then_logprob_tiebreak(
                r["sample_predictions"], r["sample_avg_logprob"], r["baseline_prediction"], m
            )
        )

    print(f"{'strategy':38s} {'correct':>9s} {'delta':>7s} {'gains':>6s} {'regressions':>11s}")
    for name, fn in strategies.items():
        correct = gains = regressions = 0
        for r in rows:
            choice = fn(r)
            before = r["baseline_prediction"] == r["answer"]
            after = choice == r["answer"]
            correct += int(after)
            if choice != r["baseline_prediction"]:
                gains += int(not before and after)
                regressions += int(before and not after)
        delta = correct - baseline_correct
        print(f"{name:38s} {correct:>6d}/{total} {delta:>+7d} {gains:>6d} {regressions:>11d}")


if __name__ == "__main__":
    main()
