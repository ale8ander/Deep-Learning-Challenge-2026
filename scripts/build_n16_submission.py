#!/usr/bin/env python3
"""Build a submission from two independent N=8 self-consistency runs.

Only IDs present in the JSONL inputs are rewritten.  For each such ID, the
sixteen normalized integer predictions are pooled.  A unique plurality is
accepted when its count reaches ``--min-count``; otherwise the original
five-voter ``baseline_prediction`` is used.
"""

from __future__ import annotations

import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Any


INTEGER_RE = re.compile(r"^[+-]?\d+$")


def normalize_integer(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not INTEGER_RE.fullmatch(text):
        return None
    return str(int(text))


def read_jsonl(path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            row = json.loads(line)
            row_id = str(row.get("id", ""))
            if not row_id:
                raise ValueError(f"{path}:{line_number}: missing id")
            if row_id in rows:
                raise ValueError(f"{path}:{line_number}: duplicate id {row_id}")
            rows[row_id] = row
    if not rows:
        raise ValueError(f"{path}: no rows")
    return rows


def parsed_samples(row: dict[str, Any], path: Path, row_id: str) -> list[str]:
    raw = row.get("sample_predictions")
    if not isinstance(raw, list) or len(raw) != 8:
        raise ValueError(
            f"{path}: {row_id} must contain exactly 8 sample_predictions; "
            f"got {len(raw) if isinstance(raw, list) else type(raw).__name__}"
        )
    samples = [normalize_integer(value) for value in raw]
    invalid = [index for index, value in enumerate(samples) if value is None]
    if invalid:
        raise ValueError(f"{path}: {row_id} has invalid samples at indexes {invalid}")
    return [value for value in samples if value is not None]


def choose_prediction(samples: list[str], baseline: str, min_count: int) -> tuple[str, int, bool]:
    counts = Counter(samples)
    top_count = max(counts.values())
    winners = [prediction for prediction, count in counts.items() if count == top_count]
    accepted = len(winners) == 1 and top_count >= min_count
    return (winners[0] if accepted else baseline), top_count, accepted


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--original", type=Path, required=True, help="Original N=8 JSONL")
    parser.add_argument("--extra", type=Path, required=True, help="Independent extra N=8 JSONL")
    parser.add_argument("--base-submission", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--min-count", type=int, default=7)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not 1 <= args.min_count <= 16:
        raise ValueError("--min-count must be between 1 and 16")

    original = read_jsonl(args.original)
    extra = read_jsonl(args.extra)
    if set(original) != set(extra):
        only_original = sorted(set(original) - set(extra))[:10]
        only_extra = sorted(set(extra) - set(original))[:10]
        raise ValueError(
            "N=8 input ID sets differ: "
            f"only_original={only_original}, only_extra={only_extra}"
        )

    selected: dict[str, str] = {}
    baselines: dict[str, str] = {}
    accepted_count = 0
    overrides_vs_baseline = 0
    top_count_histogram: Counter[int] = Counter()

    for row_id, original_row in original.items():
        extra_row = extra[row_id]
        original_baseline = normalize_integer(original_row.get("baseline_prediction"))
        extra_baseline = normalize_integer(extra_row.get("baseline_prediction"))
        if original_baseline is None or extra_baseline is None:
            raise ValueError(f"{row_id}: invalid baseline_prediction")
        if original_baseline != extra_baseline:
            raise ValueError(
                f"{row_id}: baseline mismatch ({original_baseline} != {extra_baseline})"
            )
        original_support = original_row.get("baseline_support")
        extra_support = extra_row.get("baseline_support")
        if original_support != extra_support:
            raise ValueError(f"{row_id}: baseline_support mismatch")

        samples = parsed_samples(original_row, args.original, row_id)
        samples += parsed_samples(extra_row, args.extra, row_id)
        if len(samples) != 16:
            raise AssertionError(f"{row_id}: expected 16 valid samples")
        prediction, top_count, accepted = choose_prediction(
            samples, original_baseline, args.min_count
        )
        selected[row_id] = prediction
        baselines[row_id] = original_baseline
        top_count_histogram[top_count] += 1
        accepted_count += int(accepted)
        overrides_vs_baseline += int(prediction != original_baseline)

    with args.base_submission.open(newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames != ["id", "answer"]:
            raise ValueError(
                f"{args.base_submission}: expected header id,answer; got {reader.fieldnames}"
            )
        base_rows = list(reader)

    base_ids = [row["id"] for row in base_rows]
    if len(base_ids) != len(set(base_ids)):
        raise ValueError(f"{args.base_submission}: duplicate IDs")
    missing = sorted(set(selected) - set(base_ids))
    if missing:
        raise ValueError(f"N=16 IDs absent from base submission: {missing[:10]}")

    changes_vs_base = 0
    rewritten = 0
    output_rows: list[dict[str, str]] = []
    for row in base_rows:
        row_id = row["id"]
        answer = normalize_integer(row["answer"])
        if answer is None:
            raise ValueError(f"{args.base_submission}: invalid answer for {row_id}")
        if row_id in selected:
            rewritten += 1
            changes_vs_base += int(selected[row_id] != answer)
            answer = selected[row_id]
        output_rows.append({"id": row_id, "answer": answer})

    if rewritten != len(selected):
        raise AssertionError(f"rewrote {rewritten} rows, expected {len(selected)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=["id", "answer"], lineterminator="\n")
        writer.writeheader()
        writer.writerows(output_rows)

    summary = {
        "output": str(args.output),
        "submission_rows": len(output_rows),
        "n16_ids": len(selected),
        "min_count": args.min_count,
        "accepted_unique_plurality": accepted_count,
        "overrides_vs_five_voter_baseline": overrides_vs_baseline,
        "changes_vs_base_submission": changes_vs_base,
        "top_count_histogram": dict(sorted(top_count_histogram.items())),
    }
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
