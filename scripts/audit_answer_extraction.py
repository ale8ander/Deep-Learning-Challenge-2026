import argparse
import json
import re
from collections import Counter
from pathlib import Path

from ensemble_predictions import normalize_integer


FINAL_RE = re.compile(
    r"(?:final\s+answer|최종\s*정답)\s*(?:is|:|=)?\s*"
    r"(?:\\boxed\s*\{\s*)?(-?\d[\d,]*)",
    re.IGNORECASE,
)
BOXED_RE = re.compile(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}")
ANSWER_RE = re.compile(
    r"(?:answer|정답)\s*(?:is|:|=)?\s*(-?\d[\d,]*)",
    re.IGNORECASE,
)
INTEGER_RE = re.compile(r"(-?\d[\d,]*)")


def corrected_extract(text: str) -> str | None:
    for pattern in (FINAL_RE, BOXED_RE, ANSWER_RE, INTEGER_RE):
        matches = pattern.findall(text)
        if matches:
            return normalize_integer(matches[-1].replace(",", ""))
    return None


def select(samples: list[str | None], baseline: str | None, minimum: int) -> str | None:
    counts = Counter(value for value in samples if value is not None)
    if not counts:
        return baseline
    best = max(counts.values())
    winners = [value for value, count in counts.items() if count == best]
    if len(winners) == 1 and best >= minimum:
        return winners[0]
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--min-count", type=int, default=4)
    args = parser.parse_args()

    rows = []
    with args.input.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            responses = row.get("responses")
            stored_raw = row.get("sample_predictions")
            if not isinstance(responses, list) or not isinstance(stored_raw, list):
                raise SystemExit(f"{args.input}:{line_number}: missing sample fields")
            if len(responses) != len(stored_raw):
                raise SystemExit(f"{args.input}:{line_number}: sample length mismatch")
            stored = [normalize_integer(value) for value in stored_raw]
            corrected = [corrected_extract(response) for response in responses]
            answer = normalize_integer(row.get("answer"))
            baseline = normalize_integer(row.get("baseline_prediction"))
            stored_choice = select(stored, baseline, args.min_count)
            corrected_choice = select(corrected, baseline, args.min_count)
            rows.append(
                {
                    "id": str(row.get("id")),
                    "answer": answer,
                    "baseline": baseline,
                    "stored": stored,
                    "corrected": corrected,
                    "stored_choice": stored_choice,
                    "corrected_choice": corrected_choice,
                }
            )

    sample_differences = sample_gains = sample_regressions = 0
    baseline_correct = stored_correct = corrected_correct = 0
    choice_differences = choice_gains = choice_regressions = 0
    stored_oracle = corrected_oracle = corrected_new_oracle = 0
    changed_rows = []
    for row in rows:
        answer = row["answer"]
        pairs = list(zip(row["stored"], row["corrected"], strict=True))
        sample_differences += sum(before != after for before, after in pairs)
        sample_gains += sum(before != answer and after == answer for before, after in pairs)
        sample_regressions += sum(before == answer and after != answer for before, after in pairs)
        baseline_correct += int(row["baseline"] == answer)
        stored_correct += int(row["stored_choice"] == answer)
        corrected_correct += int(row["corrected_choice"] == answer)
        changed = row["stored_choice"] != row["corrected_choice"]
        choice_differences += int(changed)
        choice_gains += int(
            changed and row["stored_choice"] != answer and row["corrected_choice"] == answer
        )
        choice_regressions += int(
            changed and row["stored_choice"] == answer and row["corrected_choice"] != answer
        )
        old_oracle = answer in row["stored"]
        new_oracle = answer in row["corrected"]
        stored_oracle += int(old_oracle)
        corrected_oracle += int(new_oracle)
        corrected_new_oracle += int(new_oracle and not old_oracle)
        if changed or any(before != after for before, after in pairs):
            changed_rows.append(row)

    print(f"problems={len(rows)} samples={sum(len(row['stored']) for row in rows)}")
    print(
        f"sample_differences={sample_differences} "
        f"sample_gains={sample_gains} sample_regressions={sample_regressions}"
    )
    print(
        f"baseline={baseline_correct}/{len(rows)} "
        f"stored_min{args.min_count}={stored_correct}/{len(rows)} "
        f"corrected_min{args.min_count}={corrected_correct}/{len(rows)}"
    )
    print(
        f"choice_differences={choice_differences} "
        f"choice_gains={choice_gains} choice_regressions={choice_regressions}"
    )
    print(
        f"stored_oracle={stored_oracle}/{len(rows)} "
        f"corrected_oracle={corrected_oracle}/{len(rows)} "
        f"corrected_new_oracle={corrected_new_oracle}"
    )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            "\n".join(json.dumps(row, ensure_ascii=False) for row in changed_rows) + "\n",
            encoding="utf-8",
        )
        print(f"changed_output={args.output} rows={len(changed_rows)}")


if __name__ == "__main__":
    main()
