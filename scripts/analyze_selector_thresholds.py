import argparse
import json
from pathlib import Path


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument(
        "--only-problem-ids-from",
        type=Path,
        help="JSONL whose problem_id or id field defines the evaluation subset.",
    )
    parser.add_argument(
        "--thresholds",
        type=float,
        nargs="+",
        default=[0.0, 0.25, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0],
    )
    parser.add_argument(
        "--require-swap-consistency",
        action=argparse.BooleanOptionalAction,
        default=True,
    )
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    allowed = None
    if args.only_problem_ids_from:
        allowed = set()
        for row in read_jsonl(args.only_problem_ids_from):
            problem_id = str(row.get("problem_id") or row.get("id") or "").strip()
            if problem_id:
                allowed.add(problem_id)
        rows = [row for row in rows if str(row.get("id", "")).strip() in allowed]
    rows = [row for row in rows if row.get("answer") is not None]
    if not rows:
        raise SystemExit("No labeled rows selected")

    baseline_correct = sum(
        str(row["fallback_prediction"]) == str(row["answer"]) for row in rows
    )
    results = []
    for threshold in sorted(set(args.thresholds)):
        selected_correct = 0
        overrides = 0
        gains = 0
        regressions = 0
        changed_wrong = 0
        for row in rows:
            fallback = str(row["fallback_prediction"])
            prediction = fallback
            best_margin = float("inf")
            for comparison in row.get("comparisons", []):
                margin = float(comparison["current_minus_alternative_margin"])
                eligible = margin < -threshold
                if args.require_swap_consistency:
                    eligible = eligible and bool(
                        comparison.get("alternative_preferred_consistently")
                    )
                if eligible and margin < best_margin:
                    best_margin = margin
                    prediction = str(comparison["alternative"])
            answer = str(row["answer"])
            before = fallback == answer
            after = prediction == answer
            changed = prediction != fallback
            selected_correct += int(after)
            overrides += int(changed)
            gains += int(changed and not before and after)
            regressions += int(changed and before and not after)
            changed_wrong += int(changed and not before and not after)
        results.append(
            {
                "threshold": threshold,
                "samples": len(rows),
                "baseline_correct": baseline_correct,
                "selected_correct": selected_correct,
                "delta": selected_correct - baseline_correct,
                "overrides": overrides,
                "gains": gains,
                "regressions": regressions,
                "changed_wrong": changed_wrong,
            }
        )
    print(json.dumps(results, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
