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


def plurality(samples: list[str | None], baseline: str | None, minimum: int) -> str | None:
    counts = Counter(value for value in samples if value is not None)
    if not counts:
        return baseline
    best = max(counts.values())
    winners = [value for value, count in counts.items() if count == best]
    if len(winners) == 1 and best >= minimum:
        return winners[0]
    return baseline


def cross_prompt(
    default: list[str | None],
    verify: list[str | None],
    baseline: str | None,
    minimum_total: int,
    minimum_each: int,
) -> str | None:
    default_counts = Counter(value for value in default if value is not None)
    verify_counts = Counter(value for value in verify if value is not None)
    eligible = {}
    for value in default_counts.keys() & verify_counts.keys():
        if (
            default_counts[value] >= minimum_each
            and verify_counts[value] >= minimum_each
            and default_counts[value] + verify_counts[value] >= minimum_total
        ):
            eligible[value] = default_counts[value] + verify_counts[value]
    if not eligible:
        return baseline
    best = max(eligible.values())
    winners = [value for value, count in eligible.items() if count == best]
    return winners[0] if len(winners) == 1 else baseline


def unique_mode(samples: list[str | None], minimum: int) -> str | None:
    counts = Counter(value for value in samples if value is not None)
    if not counts:
        return None
    best = max(counts.values())
    winners = [value for value, count in counts.items() if count == best]
    if len(winners) == 1 and best >= minimum:
        return winners[0]
    return None


def registered_mixed_rule(
    default: list[str | None],
    verify: list[str | None],
    baseline: str | None,
    rule: str,
) -> str | None:
    pooled = default + verify
    if rule == "m4":
        if sum(value is not None for value in pooled) < 6:
            return baseline
        return plurality(pooled, baseline, 4)
    if (
        sum(value is not None for value in default) < 3
        or sum(value is not None for value in verify) < 3
    ):
        return baseline
    default_mode = unique_mode(default, 2)
    verify_mode = unique_mode(verify, 2)
    if default_mode is None or default_mode != verify_mode:
        return baseline
    if rule == "x22_5" and pooled.count(default_mode) < 5:
        return baseline
    return default_mode


def summarize(rows: list[dict], prediction_key: str) -> dict:
    baseline_correct = selected_correct = overrides = gains = regressions = 0
    wrong_to_wrong = sample_oracle = expanded_oracle = 0
    for row in rows:
        baseline = row["baseline"]
        prediction = row[prediction_key]
        answer = row["answer"]
        before = baseline == answer
        after = prediction == answer
        changed = prediction != baseline
        baseline_correct += int(before)
        selected_correct += int(after)
        overrides += int(changed)
        gains += int(changed and not before and after)
        regressions += int(changed and before and not after)
        wrong_to_wrong += int(changed and not before and not after)
        oracle = answer in row["samples"][prediction_key.split(":", 1)[0]]
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
        "wrong_to_wrong": wrong_to_wrong,
        "sample_oracle": sample_oracle,
        "expanded_oracle": expanded_oracle,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--default", type=Path, required=True)
    parser.add_argument("--verify", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--split-seed", type=int, default=20260831)
    parser.add_argument("--allow-partial", action="store_true")
    args = parser.parse_args()

    default_rows = read_jsonl(args.default)
    verify_rows = read_jsonl(args.verify)
    if set(default_rows) != set(verify_rows) and not args.allow_partial:
        raise SystemExit("Default and verify ID sets differ")
    problem_ids = sorted(set(default_rows) & set(verify_rows))
    if not problem_ids:
        raise SystemExit("Default and verify have no shared IDs")

    rows = []
    for problem_id in problem_ids:
        default_row = default_rows[problem_id]
        verify_row = verify_rows[problem_id]
        default_samples = [
            normalize_integer(value) for value in default_row["sample_predictions"]
        ]
        verify_samples = [
            normalize_integer(value) for value in verify_row["sample_predictions"]
        ]
        if len(default_samples) != 8 or len(verify_samples) != 4:
            raise SystemExit(
                f"{problem_id}: expected default N=8 and verify N=4, "
                f"got {len(default_samples)} and {len(verify_samples)}"
            )
        answer = normalize_integer(default_row.get("answer"))
        if answer != normalize_integer(verify_row.get("answer")):
            raise SystemExit(f"{problem_id}: answer mismatch")
        baseline = normalize_integer(default_row.get("baseline_prediction"))
        if baseline != normalize_integer(verify_row.get("baseline_prediction")):
            raise SystemExit(f"{problem_id}: baseline mismatch")
        families = {
            "default_first4": default_samples[:4],
            "default_last4": default_samples[4:],
            "default8": default_samples,
            "verify4": verify_samples,
            "mixed": default_samples[:4] + verify_samples,
            "all12": default_samples + verify_samples,
        }
        row = {
            "id": problem_id,
            "split": split_name(problem_id, args.split_seed),
            "answer": answer,
            "baseline": baseline,
            "samples": families,
        }
        for family, samples in families.items():
            upper = len(samples) - 2 if len(samples) > 4 else 3
            for minimum in range(2, upper + 1):
                row[f"{family}:plurality_min{minimum}"] = plurality(
                    samples, baseline, minimum
                )
        for minimum_each in (1, 2):
            for minimum_total in range(max(3, 2 * minimum_each), 7):
                key = f"mixed:cross_each{minimum_each}_total{minimum_total}"
                row[key] = cross_prompt(
                    default_samples[:4],
                    verify_samples,
                    baseline,
                    minimum_total,
                    minimum_each,
                )
        for rule in ("m4", "x22", "x22_5"):
            row[f"mixed:registered_{rule}"] = registered_mixed_rule(
                default_samples[:4], verify_samples, baseline, rule
            )
        rows.append(row)

    prediction_keys = sorted(
        key for key in rows[0] if ":" in key
    )
    summaries = {}
    print(
        "split,rule,n,baseline,selected,delta,overrides,gains,"
        "regressions,wrong_to_wrong,sample_oracle,expanded_oracle"
    )
    for split in ("calibration", "validation", "all"):
        subset = rows if split == "all" else [row for row in rows if row["split"] == split]
        for key in prediction_keys:
            metrics = summarize(subset, key)
            summaries[f"{split}:{key}"] = metrics
            print(
                f"{split},{key},{metrics['n']},{metrics['baseline']},"
                f"{metrics['selected']},{metrics['delta']:+d},"
                f"{metrics['overrides']},{metrics['gains']},"
                f"{metrics['regressions']},{metrics['wrong_to_wrong']},"
                f"{metrics['sample_oracle']},{metrics['expanded_oracle']}"
            )

    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(
                {
                    "default": str(args.default),
                    "verify": str(args.verify),
                    "split_seed": args.split_seed,
                    "summaries": summaries,
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
