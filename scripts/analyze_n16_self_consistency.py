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


def select(samples: list[str | None], baseline: str | None, minimum: int) -> str | None:
    counts = Counter(value for value in samples if value is not None)
    if not counts:
        return baseline
    best = max(counts.values())
    winners = [value for value, count in counts.items() if count == best]
    if len(winners) == 1 and best >= minimum:
        return winners[0]
    return baseline


def metrics(rows: list[dict], family: str, minimum: int) -> dict:
    baseline_correct = selected_correct = overrides = gains = regressions = 0
    sample_oracle = expanded_oracle = 0
    for row in rows:
        baseline = row["baseline"]
        answer = row["answer"]
        prediction = select(row[family], baseline, minimum)
        before = baseline == answer
        after = prediction == answer
        changed = prediction != baseline
        baseline_correct += int(before)
        selected_correct += int(after)
        overrides += int(changed)
        gains += int(changed and not before and after)
        regressions += int(changed and before and not after)
        oracle = answer in row[family]
        sample_oracle += int(oracle)
        expanded_oracle += int(before or oracle)
    return {
        "n": len(rows),
        "baseline": baseline_correct,
        "selected": selected_correct,
        "delta": selected_correct - baseline_correct,
        "overrides": overrides,
        "gains": gains,
        "regressions": regressions,
        "sample_oracle": sample_oracle,
        "expanded_oracle": expanded_oracle,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--original", type=Path, required=True)
    parser.add_argument("--extra", type=Path, required=True)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--merged-output", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260902)
    parser.add_argument("--allow-partial", action="store_true")
    parser.add_argument("--oracle-only", action="store_true")
    args = parser.parse_args()

    original = read_jsonl(args.original)
    extra = read_jsonl(args.extra)
    if set(original) != set(extra) and not args.allow_partial:
        raise SystemExit("Original and extra ID sets differ")
    problem_ids = sorted(set(original) & set(extra))
    if not problem_ids:
        raise SystemExit("No shared IDs")

    rows = []
    for problem_id in problem_ids:
        old_row = original[problem_id]
        new_row = extra[problem_id]
        first8 = [normalize_integer(value) for value in old_row["sample_predictions"]]
        second8 = [normalize_integer(value) for value in new_row["sample_predictions"]]
        if len(first8) != 8 or len(second8) != 8:
            raise SystemExit(f"{problem_id}: expected 8+8 samples")
        answer = normalize_integer(old_row.get("answer"))
        baseline = normalize_integer(old_row.get("baseline_prediction"))
        if answer != normalize_integer(new_row.get("answer")):
            raise SystemExit(f"{problem_id}: answer mismatch")
        if baseline != normalize_integer(new_row.get("baseline_prediction")):
            raise SystemExit(f"{problem_id}: baseline mismatch")
        rows.append(
            {
                "id": problem_id,
                "split": split_name(problem_id, args.split_seed),
                "answer": answer,
                "baseline": baseline,
                "first8": first8,
                "second8": second8,
                "n16": first8 + second8,
            }
        )

    old_oracle = sum(row["answer"] in row["first8"] for row in rows)
    new_oracle = sum(row["answer"] in row["second8"] for row in rows)
    n16_oracle = sum(row["answer"] in row["n16"] for row in rows)
    extra_unique = sum(
        row["answer"] not in row["first8"] and row["answer"] in row["second8"]
        for row in rows
    )
    baseline_oracle = sum(row["baseline"] == row["answer"] for row in rows)
    expanded8 = sum(
        row["baseline"] == row["answer"] or row["answer"] in row["first8"]
        for row in rows
    )
    expanded16 = sum(
        row["baseline"] == row["answer"] or row["answer"] in row["n16"]
        for row in rows
    )
    print(
        f"problems={len(rows)} baseline={baseline_oracle} "
        f"first8_oracle={old_oracle} second8_oracle={new_oracle} "
        f"n16_oracle={n16_oracle} extra_unique={extra_unique} "
        f"expanded8={expanded8} expanded16={expanded16} "
        f"expanded_delta={expanded16 - expanded8:+d}"
    )
    if args.oracle_only:
        return

    summaries = {}
    print(
        "split,family,min_count,n,baseline,selected,delta,overrides,"
        "gains,regressions,sample_oracle,expanded_oracle"
    )
    for split in ("calibration", "validation", "all"):
        subset = rows if split == "all" else [row for row in rows if row["split"] == split]
        for family, thresholds in (
            ("first8", range(2, 7)),
            ("second8", range(2, 7)),
            ("n16", range(3, 13)),
        ):
            for minimum in thresholds:
                result = metrics(subset, family, minimum)
                key = f"{split}:{family}:min{minimum}"
                summaries[key] = result
                print(
                    f"{split},{family},{minimum},{result['n']},"
                    f"{result['baseline']},{result['selected']},"
                    f"{result['delta']:+d},{result['overrides']},"
                    f"{result['gains']},{result['regressions']},"
                    f"{result['sample_oracle']},{result['expanded_oracle']}"
                )

    if args.summary_output is not None:
        args.summary_output.parent.mkdir(parents=True, exist_ok=True)
        args.summary_output.write_text(
            json.dumps(
                {
                    "original": str(args.original),
                    "extra": str(args.extra),
                    "split_seed": args.split_seed,
                    "problems": len(rows),
                    "first8_oracle": old_oracle,
                    "second8_oracle": new_oracle,
                    "n16_oracle": n16_oracle,
                    "extra_unique": extra_unique,
                    "expanded8": expanded8,
                    "expanded16": expanded16,
                    "summaries": summaries,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        print(f"summary_output={args.summary_output}")

    if args.merged_output is not None:
        args.merged_output.parent.mkdir(parents=True, exist_ok=True)
        with args.merged_output.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
        print(f"merged_output={args.merged_output}")


if __name__ == "__main__":
    main()
