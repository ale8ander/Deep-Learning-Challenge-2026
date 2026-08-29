import argparse
import csv
import hashlib
import json
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ensemble_predictions import normalize_integer
from submit_baseline import SYSTEM_PROMPTS, extract_answer, response_token_count


DEFAULT_MODEL_PATH = Path("/workspace/models/Qwen2.5-3B-Instruct")


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_predictions(path: Path) -> dict[str, dict]:
    if path.suffix.lower() == ".csv":
        rows = {}
        for line_number, row in enumerate(read_csv(path), start=2):
            problem_id = str(row.get("id", "")).strip()
            prediction = row.get("prediction", row.get("answer"))
            if not problem_id or problem_id in rows:
                raise SystemExit(f"Missing or duplicate id in {path}:{line_number}")
            rows[problem_id] = {"id": problem_id, "prediction": prediction}
        return rows

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


def choose_existing(votes: list[str | None], fallback_index: int) -> tuple[str | None, int]:
    counts = Counter(votes)
    best = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == best]
    if len(winners) == 1 and best >= 2:
        return winners[0], best
    fallback = votes[fallback_index]
    return fallback, counts[fallback]


def stable_key(problem_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{problem_id}".encode()).hexdigest()


def sampled_prediction(
    counts: Counter,
    baseline: str | None,
    min_count: int,
) -> str | None:
    if not counts:
        return baseline
    best = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == best]
    if len(winners) == 1 and best >= min_count:
        return winners[0]
    return baseline


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path)
    parser.add_argument("--predictions", type=Path, action="append")
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--submission", type=Path)
    parser.add_argument("--base-submission", type=Path)
    parser.add_argument("--analyze-output", type=Path)
    parser.add_argument("--subset-size", type=int, default=50)
    parser.add_argument("--min-baseline-support", type=int)
    parser.add_argument("--max-baseline-support", type=int)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--min-count", type=int, default=4)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--fallback-voter", type=int, default=1)
    parser.add_argument("--selection-seed", type=int, default=20260828)
    parser.add_argument("--model-seed", type=int, default=20260828)
    parser.add_argument(
        "--prompt-style", choices=tuple(SYSTEM_PROMPTS), default="default"
    )
    args = parser.parse_args()

    if args.analyze_output is not None:
        analyzed = read_predictions(args.analyze_output)
        grouped = {}
        for row in analyzed.values():
            support = int(row["baseline_support"])
            grouped.setdefault(support, []).append(row)
        print("support,n,baseline,after,delta,overrides,gains,regressions,wrong_to_wrong")
        for support, rows in sorted(grouped.items()):
            baseline_correct = after_correct = overrides = gains = regressions = 0
            for row in rows:
                counts = Counter(
                    value for value in row["sample_predictions"] if value is not None
                )
                prediction = sampled_prediction(
                    counts, row["baseline_prediction"], args.min_count
                )
                before = row["baseline_prediction"] == row["answer"]
                after = prediction == row["answer"]
                changed = prediction != row["baseline_prediction"]
                baseline_correct += int(before)
                after_correct += int(after)
                overrides += int(changed)
                gains += int(changed and not before and after)
                regressions += int(changed and before and not after)
            wrong_to_wrong = overrides - gains - regressions
            print(
                f"{support},{len(rows)},{baseline_correct},{after_correct},"
                f"{after_correct - baseline_correct:+d},{overrides},{gains},"
                f"{regressions},{wrong_to_wrong}"
            )
        return

    if not all((args.questions, args.predictions, args.adapter_path, args.output)):
        raise SystemExit(
            "--questions, --predictions, --adapter-path, and --output are required"
        )
    if len(args.predictions) < 1:
        raise SystemExit("At least one --predictions file is required")
    if args.subset_size <= 0 or args.num_samples <= 1 or args.batch_size <= 0:
        raise SystemExit("subset size and batch size must be positive; samples must exceed 1")
    if not 2 <= args.min_count <= args.num_samples:
        raise SystemExit("--min-count must be between 2 and --num-samples")
    if not 0 < args.temperature or not 0 < args.top_p <= 1:
        raise SystemExit("Invalid temperature or top-p")

    questions = read_csv(args.questions)
    if not questions or not {"id", "question"}.issubset(questions[0]):
        raise SystemExit("Questions CSV must contain id and question")
    has_ground_truth = "answer" in questions[0]
    mappings = [read_predictions(path) for path in args.predictions]
    question_ids = [row["id"].strip() for row in questions]
    expected = set(question_ids)
    for path, mapping in zip(args.predictions, mappings, strict=True):
        if set(mapping) != expected:
            raise SystemExit(f"ID set mismatch in {path}")
    fallback_index = args.fallback_voter - 1
    if not 0 <= fallback_index < len(mappings):
        raise SystemExit("--fallback-voter is outside the voter range")

    prepared = []
    for row in questions:
        problem_id = row["id"].strip()
        answer = normalize_integer(row["answer"]) if has_ground_truth else None
        votes = [normalize_integer(mapping[problem_id].get("prediction")) for mapping in mappings]
        baseline, support = choose_existing(votes, fallback_index)
        prepared.append(
            {
                "id": problem_id,
                "question": row["question"],
                "answer": answer,
                "votes": votes,
                "baseline_prediction": baseline,
                "baseline_support": support,
            }
        )
    prepared.sort(key=lambda row: (row["baseline_support"], stable_key(row["id"], args.selection_seed)))
    eligible = prepared
    if args.min_baseline_support is not None:
        eligible = [
            row for row in eligible
            if row["baseline_support"] >= args.min_baseline_support
        ]
    if args.max_baseline_support is not None:
        eligible = [
            row for row in eligible
            if row["baseline_support"] <= args.max_baseline_support
        ]
    selected = eligible[: min(args.subset_size, len(eligible))]

    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, local_files_only=True
    )
    model = PeftModel.from_pretrained(
        model, args.adapter_path, local_files_only=True
    ).cuda().eval()

    system_prompt = SYSTEM_PROMPTS[args.prompt_style]
    results = []
    started = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output, torch.inference_mode():
        for start in tqdm(range(0, len(selected), args.batch_size), desc="self-consistency"):
            batch = selected[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": row["question"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            decoded = tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            for batch_index, row in enumerate(batch):
                responses = decoded[
                    batch_index * args.num_samples : (batch_index + 1) * args.num_samples
                ]
                predictions = [normalize_integer(extract_answer(response)) for response in responses]
                valid = [prediction for prediction in predictions if prediction is not None]
                counts = Counter(valid)
                prediction = sampled_prediction(
                    counts, row["baseline_prediction"], args.min_count
                )
                answer = row["answer"]
                existing_oracle = answer in row["votes"] if has_ground_truth else None
                any_sample_correct = answer in valid if has_ground_truth else None
                result = {
                    **row,
                    "sample_predictions": predictions,
                    "sample_counts": dict(counts.most_common()),
                    "valid_samples": len(valid),
                    "prediction": prediction,
                    f"sample_prediction_min{args.min_count}": prediction,
                    "sample_correct": prediction == answer if has_ground_truth else None,
                    "baseline_correct": (
                        row["baseline_prediction"] == answer if has_ground_truth else None
                    ),
                    "existing_oracle_correct": existing_oracle,
                    "any_sample_correct": any_sample_correct,
                    "new_oracle_correct": (
                        any_sample_correct and not existing_oracle
                        if has_ground_truth else None
                    ),
                    "responses": responses,
                    "response_tokens": [
                        response_token_count(tokenizer, response) for response in responses
                    ],
                }
                results.append(result)
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()

    elapsed = time.time() - started
    print(f"samples={len(results)} generations={len(results) * args.num_samples}")
    print(f"elapsed_seconds={elapsed:.1f}")
    if has_ground_truth:
        baseline_correct = sum(row["baseline_correct"] for row in results)
        existing_oracle = sum(row["existing_oracle_correct"] for row in results)
        sample_oracle = sum(row["any_sample_correct"] for row in results)
        expanded_oracle = sum(
            row["existing_oracle_correct"] or row["any_sample_correct"] for row in results
        )
        print(
            f"baseline={baseline_correct}/{len(results)} "
            f"existing_oracle={existing_oracle}/{len(results)}"
        )
        print(
            f"sample_oracle={sample_oracle}/{len(results)} "
            f"expanded_oracle={expanded_oracle}/{len(results)} "
            f"new_oracle={sum(row['new_oracle_correct'] for row in results)}"
        )
        for minimum in (2, 3, 4, 5, 6):
            selected_correct = overrides = gains = regressions = 0
            for row in results:
                counts = Counter(
                    value for value in row["sample_predictions"] if value is not None
                )
                prediction = sampled_prediction(
                    counts, row["baseline_prediction"], minimum
                )
                before = row["baseline_prediction"] == row["answer"]
                after = prediction == row["answer"]
                changed = prediction != row["baseline_prediction"]
                selected_correct += int(after)
                overrides += int(changed)
                gains += int(changed and not before and after)
                regressions += int(changed and before and not after)
            print(
                f"min_count={minimum} selected={selected_correct}/{len(results)} "
                f"delta={selected_correct - baseline_correct:+d} overrides={overrides} "
                f"gains={gains} regressions={regressions}"
            )

    result_by_id = {row["id"]: row for row in results}
    prepared_by_id = {row["id"]: row for row in prepared}
    preserved = None
    if args.base_submission is not None:
        preserved = read_predictions(args.base_submission)
        if set(preserved) != expected:
            raise SystemExit(f"ID set mismatch in {args.base_submission}")
    final_predictions = {
        problem_id: (
            result_by_id[problem_id]["prediction"]
            if problem_id in result_by_id
            else (
                normalize_integer(preserved[problem_id].get("prediction"))
                if preserved is not None
                else prepared_by_id[problem_id]["baseline_prediction"]
            )
        )
        for problem_id in question_ids
    }
    overrides = sum(
        final_predictions[problem_id]
        != prepared_by_id[problem_id]["baseline_prediction"]
        for problem_id in question_ids
    )
    print(f"eligible={len(eligible)} selected={len(selected)} overrides={overrides}")

    if args.submission is not None:
        missing = [key for key, value in final_predictions.items() if value is None]
        if missing:
            raise SystemExit(f"Missing final predictions for {len(missing)} ids")
        args.submission.parent.mkdir(parents=True, exist_ok=True)
        with args.submission.open("w", encoding="utf-8", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=["id", "answer"])
            writer.writeheader()
            writer.writerows(
                {"id": problem_id, "answer": final_predictions[problem_id]}
                for problem_id in question_ids
            )
        print(f"submission={args.submission} rows={len(question_ids)}")

    print(f"output={args.output}")


if __name__ == "__main__":
    import sys

    utility = sys.argv[1] if len(sys.argv) > 1 else None
    if utility in {
        "build-verifier",
        "train-verifier",
        "eval-verifier",
        "score-verifier",
        "select-verifier",
        "mixed-analyze",
        "audit-extraction",
        "rescue-truncated",
        "ensemble",
        "n16-analyze",
        "n16-support",
        "n16-submission",
        "extend-samples",
    }:
        sys.argv.pop(1)
        if utility == "build-verifier":
            from build_stochastic_verifier_data import main as utility_main
        elif utility == "train-verifier":
            from train_qlora import main as utility_main
        elif utility == "eval-verifier":
            from evaluate_pairwise_selector import main as utility_main
        elif utility == "score-verifier":
            from score_stochastic_verifier import main as utility_main
        elif utility == "mixed-analyze":
            from analyze_mixed_self_consistency import main as utility_main
        elif utility == "audit-extraction":
            from audit_answer_extraction import main as utility_main
        elif utility == "rescue-truncated":
            from rescue_truncated_predictions import main as utility_main
        elif utility == "ensemble":
            from ensemble_predictions import main as utility_main
        elif utility == "n16-analyze":
            from analyze_n16_self_consistency import main as utility_main
        elif utility == "n16-support":
            from analyze_n16_by_support import main as utility_main
        elif utility == "n16-submission":
            from build_n16_submission import main as utility_main
        elif utility == "extend-samples":
            from extend_self_consistency_samples import main as utility_main
        else:
            from select_with_stochastic_verifier import main as utility_main
        utility_main()
    else:
        main()
