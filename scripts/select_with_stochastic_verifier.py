import argparse
import hashlib
import json
import math
from pathlib import Path


STATISTICS = ("mean", "top2_mean", "max")
MIN_SAMPLE_COUNTS = (1, 2, 3, 4)
COUNT_WEIGHTS = (0.0, 0.25, 0.5, 1.0, 2.0)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def stable_key(seed: int, problem_id: str) -> str:
    return hashlib.sha256(f"{seed}:{problem_id}".encode()).hexdigest()


def is_calibration(problem_id: str, seed: int, fraction: float) -> bool:
    value = int(stable_key(seed, problem_id)[:16], 16)
    return value / 2**64 < fraction


def statistic(values: list[float], name: str) -> float:
    if name == "mean":
        return sum(values) / len(values)
    if name == "max":
        return max(values)
    if name == "top2_mean":
        selected = sorted(values, reverse=True)[:2]
        return sum(selected) / len(selected)
    raise ValueError(name)


def proposal(row: dict, config: dict) -> tuple[str, float, dict] | None:
    baseline = row["baseline_prediction"]
    grouped: dict[str, dict] = {}
    for item in row["items"]:
        prediction = item["prediction"]
        group = grouped.setdefault(prediction, {"margins": [], "sample_count": 0})
        group["margins"].append(float(item["a_minus_b_logit"]))
        group["sample_count"] += int(item["source"] == "sample")
    if baseline not in grouped:
        raise SystemExit(f"Baseline group missing for {row['id']}")
    for group in grouped.values():
        group["raw_score"] = statistic(group["margins"], config["statistic"])
        group["score"] = group["raw_score"] + config["count_weight"] * math.log1p(
            group["sample_count"]
        )
    baseline_score = grouped[baseline]["score"]
    eligible = [
        (prediction, group)
        for prediction, group in grouped.items()
        if prediction != baseline
        and group["sample_count"] >= config["min_sample_count"]
    ]
    if not eligible:
        return None
    eligible.sort(key=lambda item: (item[1]["score"], item[1]["sample_count"], item[0]))
    prediction, group = eligible[-1]
    details = {
        "baseline_score": baseline_score,
        "candidate_score": group["score"],
        "candidate_sample_count": group["sample_count"],
        "candidate_raw_score": group["raw_score"],
    }
    return prediction, group["score"] - baseline_score, details


def evaluate(rows: list[dict], config: dict, threshold: float) -> dict:
    baseline_correct = selected_correct = overrides = gains = regressions = wrong_to_wrong = 0
    decisions = []
    for row in rows:
        answer = row.get("answer")
        if answer is None:
            raise SystemExit("Labeled rows are required for calibration/evaluation")
        baseline = row["baseline_prediction"]
        proposed = proposal(row, config)
        prediction = baseline
        difference = None
        details = {}
        if proposed is not None:
            candidate, difference, details = proposed
            if difference >= threshold:
                prediction = candidate
        before = baseline == answer
        after = prediction == answer
        changed = prediction != baseline
        baseline_correct += int(before)
        selected_correct += int(after)
        overrides += int(changed)
        gains += int(changed and not before and after)
        regressions += int(changed and before and not after)
        wrong_to_wrong += int(changed and not before and not after)
        decisions.append(
            {
                "id": row["id"],
                "answer": answer,
                "baseline_prediction": baseline,
                "prediction": prediction,
                "changed": changed,
                "before_correct": before,
                "after_correct": after,
                "score_difference": difference,
                **details,
            }
        )
    return {
        "samples": len(rows),
        "baseline_correct": baseline_correct,
        "selected_correct": selected_correct,
        "delta": selected_correct - baseline_correct,
        "overrides": overrides,
        "gains": gains,
        "regressions": regressions,
        "wrong_to_wrong": wrong_to_wrong,
        "decisions": decisions,
    }


def calibrate(
    rows: list[dict], min_overrides: int, min_gain_ratio: float
) -> tuple[dict, dict]:
    best_valid = None
    best_any = None
    for statistic_name in STATISTICS:
        for min_count in MIN_SAMPLE_COUNTS:
            for count_weight in COUNT_WEIGHTS:
                base_config = {
                    "statistic": statistic_name,
                    "min_sample_count": min_count,
                    "count_weight": count_weight,
                }
                differences = []
                for row in rows:
                    candidate = proposal(row, base_config)
                    if candidate is not None:
                        differences.append(candidate[1])
                thresholds = sorted(set(differences))
                if not thresholds:
                    continue
                for threshold in thresholds:
                    metrics = evaluate(rows, base_config, threshold)
                    key = (
                        metrics["selected_correct"],
                        -metrics["regressions"],
                        -metrics["overrides"],
                        threshold,
                    )
                    record = (key, {**base_config, "threshold": threshold}, metrics)
                    if best_any is None or key > best_any[0]:
                        best_any = record
                    valid = (
                        metrics["overrides"] >= min_overrides
                        and metrics["gains"]
                        >= min_gain_ratio * metrics["regressions"]
                        and metrics["delta"] > 0
                    )
                    if valid and (best_valid is None or key > best_valid[0]):
                        best_valid = record
    if best_any is None:
        raise SystemExit("No verifier proposals available for calibration")
    selected = best_valid if best_valid is not None else best_any
    config = {
        **selected[1],
        "passes_calibration_constraints": best_valid is not None,
        "min_overrides_constraint": min_overrides,
        "min_gain_ratio_constraint": min_gain_ratio,
    }
    return config, selected[2]


def write_decisions(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def metric_summary(metrics: dict) -> dict:
    return {key: value for key, value in metrics.items() if key != "decisions"}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--scored", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--config-output", type=Path)
    parser.add_argument("--config-input", type=Path)
    parser.add_argument("--calibration-seed", type=int, default=20260830)
    parser.add_argument("--calibration-fraction", type=float, default=0.2)
    parser.add_argument("--min-calibration-overrides", type=int, default=10)
    parser.add_argument("--min-gain-ratio", type=float, default=2.0)
    args = parser.parse_args()

    rows = read_jsonl(args.scored)
    if not rows:
        raise SystemExit("Empty scored data")
    if args.config_input is None:
        if not 0 < args.calibration_fraction < 1:
            raise SystemExit("--calibration-fraction must be between 0 and 1")
        calibration_rows = [
            row
            for row in rows
            if is_calibration(row["id"], args.calibration_seed, args.calibration_fraction)
        ]
        config, calibration_metrics = calibrate(
            calibration_rows,
            args.min_calibration_overrides,
            args.min_gain_ratio,
        )
        config["calibration_seed"] = args.calibration_seed
        config["calibration_fraction"] = args.calibration_fraction
        config["calibration_metrics"] = metric_summary(calibration_metrics)
        if args.config_output is None:
            raise SystemExit("--config-output is required during calibration")
        args.config_output.parent.mkdir(parents=True, exist_ok=True)
        args.config_output.write_text(
            json.dumps(config, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
        )
        evaluated_rows = [
            row
            for row in rows
            if not is_calibration(row["id"], args.calibration_seed, args.calibration_fraction)
        ]
        print(f"calibration={json.dumps(metric_summary(calibration_metrics))}")
    else:
        config = json.loads(args.config_input.read_text(encoding="utf-8"))
        evaluated_rows = rows

    metrics = evaluate(evaluated_rows, config, float(config["threshold"]))
    write_decisions(args.output, metrics["decisions"])
    print(f"config={json.dumps({k: v for k, v in config.items() if k != 'calibration_metrics'})}")
    print(f"evaluation={json.dumps(metric_summary(metrics))}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
