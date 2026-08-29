import argparse
import csv
import hashlib
import json
from pathlib import Path

from ensemble_predictions import normalize_integer


SYSTEM_PROMPT = (
    "Judge whether the candidate solution's submitted final integer is correct for "
    "the problem. Check the reasoning and arithmetic. Reply exactly A for correct "
    "or B for incorrect."
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


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


def stable_key(seed: int, *parts: object) -> str:
    payload = ":".join(str(part) for part in (seed, *parts))
    return hashlib.sha256(payload.encode()).hexdigest()


def is_dev(problem_id: str, seed: int, fraction: float) -> bool:
    value = int(stable_key(seed, problem_id)[:16], 16)
    return value / 2**64 < fraction


def shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 3 // 4
    tail = max_chars - head
    return text[:head] + "\n[...middle omitted...]\n" + text[-tail:]


def make_record(
    problem_id: str,
    question: str,
    source: str,
    candidate_index: int,
    prediction: str,
    response: str,
    label: str,
) -> dict:
    user = (
        f"Problem:\n{question}\n\n"
        f"Candidate solution:\n{response}\n\n"
        f"Candidate final answer: {prediction}\n\n"
        "Is the submitted final integer correct? Reply exactly A or B."
    )
    return {
        "id": f"outcome:{problem_id}:{source}:{candidate_index}",
        "problem_id": problem_id,
        "source": source,
        "candidate_index": candidate_index,
        "candidate_prediction": prediction,
        "label": label,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user},
            {"role": "assistant", "content": label},
        ],
    }


def write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--baseline", type=Path)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--dev-output", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.2)
    parser.add_argument("--max-per-class", type=int, default=4)
    parser.add_argument("--max-response-chars", type=int, default=4000)
    parser.add_argument("--max-sample-tokens", type=int, default=765)
    parser.add_argument("--expected-samples", type=int, default=8)
    parser.add_argument("--seed", type=int, default=20260830)
    args = parser.parse_args()

    if not 0 < args.dev_fraction < 1:
        raise SystemExit("--dev-fraction must be between 0 and 1")
    if args.max_per_class <= 0 or args.expected_samples <= 1:
        raise SystemExit("Class cap must be positive and expected samples must exceed 1")

    questions = read_csv(args.questions)
    if not questions or not {"id", "question", "answer"}.issubset(questions[0]):
        raise SystemExit("Questions CSV must contain id, question, answer")
    question_ids = [row["id"].strip() for row in questions]
    expected = set(question_ids)
    if len(expected) != len(question_ids):
        raise SystemExit("Duplicate question ids")
    samples = read_jsonl(args.samples)
    if set(samples) != expected:
        raise SystemExit(f"ID set mismatch in {args.samples}")
    baseline = read_jsonl(args.baseline) if args.baseline is not None else None
    if baseline is not None and set(baseline) != expected:
        raise SystemExit(f"ID set mismatch in {args.baseline}")

    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    useful = truncated = invalid = 0
    train_problems = set()
    dev_problems = set()
    pass_rate_histogram = {str(value): 0 for value in range(args.expected_samples + 1)}

    for question_row in questions:
        problem_id = question_row["id"].strip()
        answer = normalize_integer(question_row["answer"])
        sample_row = samples[problem_id]
        responses = sample_row.get("responses")
        predictions = sample_row.get("sample_predictions")
        token_counts = sample_row.get("response_tokens")
        if not all(isinstance(value, list) for value in (responses, predictions, token_counts)):
            raise SystemExit(f"Missing sample lists for {problem_id}")
        if not (
            len(responses) == len(predictions) == len(token_counts) == args.expected_samples
        ):
            raise SystemExit(f"Expected {args.expected_samples} samples for {problem_id}")

        pass_count = sum(normalize_integer(value) == answer for value in predictions)
        pass_rate_histogram[str(pass_count)] += 1
        candidates = []
        for index, (response, prediction, tokens) in enumerate(
            zip(responses, predictions, token_counts, strict=True)
        ):
            normalized = normalize_integer(prediction)
            if normalized is None or not isinstance(response, str):
                invalid += 1
                continue
            if int(tokens) > args.max_sample_tokens:
                truncated += 1
                continue
            candidates.append(("sample", index, normalized, response))

        if baseline is not None:
            row = baseline[problem_id]
            normalized = normalize_integer(row.get("prediction"))
            response = row.get("response")
            if normalized is not None and isinstance(response, str):
                candidates.append(("baseline", 0, normalized, response))

        deduplicated = []
        seen = set()
        for source, index, prediction, response in candidates:
            signature = (prediction, response)
            if signature in seen:
                continue
            seen.add(signature)
            deduplicated.append((source, index, prediction, response))
        correct = [item for item in deduplicated if item[2] == answer]
        wrong = [item for item in deduplicated if item[2] != answer]
        count = min(len(correct), len(wrong), args.max_per_class)
        if count == 0:
            continue
        useful += 1
        correct.sort(key=lambda item: stable_key(args.seed, problem_id, "A", item[0], item[1]))
        wrong.sort(key=lambda item: stable_key(args.seed, problem_id, "B", item[0], item[1]))
        target = dev_rows if is_dev(problem_id, args.seed, args.dev_fraction) else train_rows
        problem_set = dev_problems if target is dev_rows else train_problems
        problem_set.add(problem_id)
        for label, items in (("A", correct[:count]), ("B", wrong[:count])):
            for source, index, prediction, response in items:
                target.append(
                    make_record(
                        problem_id,
                        question_row["question"],
                        source,
                        index,
                        prediction,
                        shorten(response, args.max_response_chars),
                        label,
                    )
                )

    if not train_rows or not dev_rows:
        raise SystemExit("Empty train or dev verifier split")
    train_rows.sort(key=lambda row: stable_key(args.seed, row["id"]))
    dev_rows.sort(key=lambda row: stable_key(args.seed, row["id"]))
    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.dev_output, dev_rows)
    manifest = {
        "questions": str(args.questions),
        "samples": str(args.samples),
        "baseline": str(args.baseline) if args.baseline is not None else None,
        "source_problems": len(questions),
        "useful_mixed_problems": useful,
        "train_problems": len(train_problems),
        "dev_problems": len(dev_problems),
        "problem_overlap": len(train_problems & dev_problems),
        "train_rows": len(train_rows),
        "dev_rows": len(dev_rows),
        "a_labels": sum(row["label"] == "A" for row in train_rows + dev_rows),
        "b_labels": sum(row["label"] == "B" for row in train_rows + dev_rows),
        "excluded_truncated_samples": truncated,
        "excluded_invalid_samples": invalid,
        "sample_pass_count_histogram": pass_rate_histogram,
        "dev_fraction": args.dev_fraction,
        "max_per_class": args.max_per_class,
        "max_response_chars": args.max_response_chars,
        "max_sample_tokens": args.max_sample_tokens,
        "expected_samples": args.expected_samples,
        "seed": args.seed,
        "train_output": str(args.train_output),
        "train_sha256": sha256(args.train_output),
        "dev_output": str(args.dev_output),
        "dev_sha256": sha256(args.dev_output),
    }
    manifest_path = args.train_output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
