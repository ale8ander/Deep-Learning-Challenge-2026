import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

from ensemble_predictions import normalize_integer


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = str(row.get("id", "")).strip()
            if not problem_id or problem_id in rows:
                raise SystemExit(f"Missing or duplicate id in {path}:{line_number}")
            rows[problem_id] = row
    return rows


def split_name(problem_id: str, seed: int) -> str:
    digest = hashlib.sha256(f"{seed}:{problem_id}".encode()).hexdigest()
    return "calibration" if int(digest[:16], 16) % 2 == 0 else "validation"


def choose_existing(votes: list[str | None], fallback_index: int = 0) -> tuple[str | None, int]:
    counts = Counter(votes)
    best = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == best]
    if len(winners) == 1 and best >= 2:
        return winners[0], best
    fallback = votes[fallback_index]
    return fallback, counts[fallback]


def sampled(samples: list[str | None], baseline: str | None, minimum: int) -> str | None:
    counts = Counter(value for value in samples if value is not None)
    if not counts:
        return baseline
    best = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == best]
    if len(winners) == 1 and best >= minimum:
        return winners[0]
    return baseline


def summarize(rows: list[dict], predict) -> dict:
    baseline_correct = selected_correct = overrides = gains = regressions = 0
    wrong_to_wrong = 0
    for row in rows:
        prediction = predict(row)
        before = row["baseline"] == row["answer"]
        after = prediction == row["answer"]
        changed = prediction != row["baseline"]
        baseline_correct += int(before)
        selected_correct += int(after)
        overrides += int(changed)
        gains += int(changed and not before and after)
        regressions += int(changed and before and not after)
        wrong_to_wrong += int(changed and not before and not after)
    return {
        "n": len(rows),
        "baseline": baseline_correct,
        "selected": selected_correct,
        "delta": selected_correct - baseline_correct,
        "overrides": overrides,
        "gains": gains,
        "regressions": regressions,
        "wrong_to_wrong": wrong_to_wrong,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--voter", type=Path, action="append", required=True)
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260903)
    args = parser.parse_args()
    if len(args.voter) != 5:
        raise SystemExit("Exactly five --voter files are required")

    voters = [read_jsonl(path) for path in args.voter]
    original = read_jsonl(args.original)
    extra = read_jsonl(args.extra)
    expected = set(original)
    for path, mapping in zip(args.voter, voters, strict=True):
        if set(mapping) != expected:
            raise SystemExit(f"ID mismatch in voter {path}")
    if set(extra) != expected:
        raise SystemExit("Extra sample ID mismatch")

    rows = []
    for problem_id in sorted(expected):
        votes = [
            normalize_integer(mapping[problem_id].get("prediction"))
            for mapping in voters
        ]
        baseline, support = choose_existing(votes)
        answer = normalize_integer(original[problem_id].get("answer"))
        first8 = [
            normalize_integer(value)
            for value in original[problem_id]["sample_predictions"]
        ]
        second8 = [
            normalize_integer(value)
            for value in extra[problem_id]["sample_predictions"]
        ]
        if len(first8) != 8 or len(second8) != 8:
            raise SystemExit(f"{problem_id}: expected 8+8 samples")
        if any(
            normalize_integer(mapping[problem_id].get("answer")) != answer
            for mapping in voters
        ):
            raise SystemExit(f"{problem_id}: official answer mismatch")
        rows.append(
            {
                "id": problem_id,
                "split": split_name(problem_id, args.split_seed),
                "answer": answer,
                "votes": votes,
                "baseline": baseline,
                "support": support,
                "first8": first8,
                "n16": first8 + second8,
            }
        )

    print(
        "split,support,family,min_count,n,baseline,selected,delta,"
        "overrides,gains,regressions,wrong_to_wrong"
    )
    tables = {}
    for split in ("calibration", "validation", "all"):
        split_rows = rows if split == "all" else [row for row in rows if row["split"] == split]
        for support in range(1, 6):
            bucket = [row for row in split_rows if row["support"] == support]
            if not bucket:
                continue
            for family, thresholds in (
                ("first8", range(2, 7)),
                ("n16", range(3, 13)),
            ):
                for minimum in thresholds:
                    result = summarize(
                        bucket,
                        lambda row, family=family, minimum=minimum: sampled(
                            row[family], row["baseline"], minimum
                        ),
                    )
                    key = f"{split}:support{support}:{family}:min{minimum}"
                    tables[key] = result
                    print(
                        f"{split},{support},{family},{minimum},{result['n']},"
                        f"{result['baseline']},{result['selected']},"
                        f"{result['delta']:+d},{result['overrides']},"
                        f"{result['gains']},{result['regressions']},"
                        f"{result['wrong_to_wrong']}"
                    )

    locked = {}
    calibration_rows = [row for row in rows if row["split"] == "calibration"]
    for support in (3, 4, 5):
        bucket = [row for row in calibration_rows if row["support"] == support]
        candidates = []
        for minimum in range(3, 13):
            result = summarize(
                bucket,
                lambda row, minimum=minimum: sampled(
                    row["n16"], row["baseline"], minimum
                ),
            )
            if (
                result["delta"] > 0
                and result["overrides"] >= 2
                and result["gains"] >= 2 * result["regressions"]
            ):
                candidates.append((result["delta"], -result["regressions"], minimum, result))
        if candidates:
            _, _, minimum, result = max(candidates)
            locked[str(support)] = {"min_count": minimum, "calibration": result}

    def current_policy(row: dict) -> str | None:
        if row["support"] == 4:
            return sampled(row["first8"], row["baseline"], 4)
        return row["baseline"]

    def locked_policy(row: dict) -> str | None:
        config = locked.get(str(row["support"]))
        if config is None:
            return row["baseline"]
        return sampled(row["n16"], row["baseline"], config["min_count"])

    policy_results = {}
    print("policy,split,n,baseline,selected,delta,overrides,gains,regressions,wrong_to_wrong")
    for split in ("calibration", "validation", "all"):
        subset = rows if split == "all" else [row for row in rows if row["split"] == split]
        for name, predict in (("current_n8_support4", current_policy), ("locked_n16", locked_policy)):
            result = summarize(subset, predict)
            policy_results[f"{name}:{split}"] = result
            print(
                f"{name},{split},{result['n']},{result['baseline']},"
                f"{result['selected']},{result['delta']:+d},"
                f"{result['overrides']},{result['gains']},"
                f"{result['regressions']},{result['wrong_to_wrong']}"
            )
    print(f"locked_thresholds={json.dumps(locked, ensure_ascii=False)}")

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "voters": [str(path) for path in args.voter],
                    "original": str(args.original),
                    "extra": str(args.extra),
                    "split_seed": args.split_seed,
                    "locked": locked,
                    "tables": tables,
                    "policy_results": policy_results,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"output={args.output}")


if __name__ == "__main__":
    main()
