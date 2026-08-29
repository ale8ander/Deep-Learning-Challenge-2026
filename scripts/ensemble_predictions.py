import argparse
import csv
import json
import re
from collections import Counter
from pathlib import Path
from typing import Optional


INTEGER_PATTERN = re.compile(r"^-?\d+$")


def normalize_integer(value: object) -> Optional[str]:
    if value is None:
        return None
    text = str(value).strip().replace(",", "")
    if not INTEGER_PATTERN.fullmatch(text):
        return None
    negative = text.startswith("-")
    digits = text[1:] if negative else text
    digits = digits.lstrip("0") or "0"
    if digits == "0":
        return "0"
    return f"-{digits}" if negative else digits


def load_prediction_jsonl(path: Path) -> tuple[list[str], dict[str, dict]]:
    order: list[str] = []
    rows: dict[str, dict] = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"Invalid JSON in {path}:{line_number}: {error}"
                ) from error
            problem_id = row.get("id")
            answer = normalize_integer(row.get("answer"))
            prediction = normalize_integer(row.get("prediction"))
            if not isinstance(problem_id, str) or not problem_id:
                raise SystemExit(f"Missing id in {path}:{line_number}")
            if problem_id in rows:
                raise SystemExit(f"Duplicate id in {path}: {problem_id}")
            if answer is None:
                raise SystemExit(f"Invalid answer in {path}:{line_number}")
            order.append(problem_id)
            rows[problem_id] = {
                "answer": answer,
                "prediction": prediction,
                "response_tokens": row.get("response_tokens"),
                "retried_truncated": bool(row.get("retried_truncated", False)),
            }
    if not rows:
        raise SystemExit(f"No predictions found in {path}")
    return order, rows


def load_submission_csv(path: Path) -> tuple[list[str], dict[str, str]]:
    order: list[str] = []
    answers: dict[str, str] = {}
    with path.open(encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        if reader.fieldnames is None or not {"id", "answer"}.issubset(reader.fieldnames):
            raise SystemExit(f"Expected id,answer columns in {path}")
        for line_number, row in enumerate(reader, start=2):
            problem_id = row.get("id", "").strip()
            answer = normalize_integer(row.get("answer"))
            if not problem_id:
                raise SystemExit(f"Missing id in {path}:{line_number}")
            if problem_id in answers:
                raise SystemExit(f"Duplicate id in {path}: {problem_id}")
            if answer is None:
                raise SystemExit(f"Invalid integer answer in {path}:{line_number}")
            order.append(problem_id)
            answers[problem_id] = answer
    if not answers:
        raise SystemExit(f"No submission rows found in {path}")
    return order, answers


def validate_id_sets(
    paths: list[Path], orders: list[list[str]], mappings: list[dict]
) -> list[str]:
    reference_ids = set(mappings[0])
    for path, mapping in zip(paths[1:], mappings[1:]):
        missing = sorted(reference_ids - set(mapping))
        extra = sorted(set(mapping) - reference_ids)
        if missing or extra:
            raise SystemExit(
                f"ID mismatch in {path}: missing={missing[:10]} extra={extra[:10]}"
            )
    if any(order != orders[0] for order in orders[1:]):
        print("warning=Input row order differs; using the first input order.")
    return orders[0]


def choose_vote(
    votes: list[Optional[str]],
    fallback_index: int,
    ignore_none: bool = False,
) -> tuple[Optional[str], str]:
    counted_votes = [vote for vote in votes if vote is not None] if ignore_none else votes
    if not counted_votes:
        return None, "fallback"
    counts = Counter(counted_votes)
    highest_count = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == highest_count]
    if len(winners) == 1 and highest_count >= 2:
        return winners[0], "majority"
    fallback = votes[fallback_index]
    if ignore_none and fallback is None:
        fallback = next((vote for vote in votes if vote is not None), None)
    return fallback, "fallback"


def evaluate(args: argparse.Namespace) -> None:
    loaded = [load_prediction_jsonl(path) for path in args.inputs]
    orders = [item[0] for item in loaded]
    mappings = [item[1] for item in loaded]
    order = validate_id_sets(args.inputs, orders, mappings)
    fallback_index = args.fallback_voter - 1
    if not 0 <= fallback_index < len(mappings):
        raise SystemExit("--fallback-voter is outside the input voter range")

    for problem_id in order:
        answers = {mapping[problem_id]["answer"] for mapping in mappings}
        if len(answers) != 1:
            raise SystemExit(f"Official answer mismatch for {problem_id}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    majority_resolved = 0
    fallback_used = 0
    changed_from_fallback = 0
    abstained_total = 0
    voter_correct = [0] * len(mappings)

    with args.output.open("w", encoding="utf-8") as output:
        for problem_id in order:
            answer = mappings[0][problem_id]["answer"]
            raw_votes = [mapping[problem_id]["prediction"] for mapping in mappings]
            abstained_voters = []
            votes = list(raw_votes)
            if args.abstain_truncated:
                for index, mapping in enumerate(mappings):
                    row = mapping[problem_id]
                    response_tokens = row.get("response_tokens")
                    if (
                        row.get("retried_truncated")
                        and isinstance(response_tokens, int)
                        and response_tokens >= args.stuck_threshold
                    ):
                        votes[index] = None
                        abstained_voters.append(index + 1)
            abstained_total += len(abstained_voters)
            prediction, decision = choose_vote(
                votes, fallback_index, ignore_none=args.abstain_truncated
            )
            is_correct = prediction == answer
            correct += int(is_correct)
            majority_resolved += int(decision == "majority")
            fallback_used += int(decision == "fallback")
            changed_from_fallback += int(prediction != raw_votes[fallback_index])
            for index, vote in enumerate(raw_votes):
                voter_correct[index] += int(vote == answer)
            output.write(
                json.dumps(
                    {
                        "id": problem_id,
                        "answer": answer,
                        "prediction": prediction,
                        "correct": is_correct,
                        "votes": votes,
                        "raw_votes": raw_votes,
                        "abstained_voters": abstained_voters,
                        "decision": decision,
                        "fallback_voter": args.fallback_voter,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )

    total = len(order)
    print(f"samples={total}")
    for index, (path, count) in enumerate(zip(args.inputs, voter_correct), start=1):
        print(f"voter{index}={count}/{total} ({count / total:.6f}) path={path}")
    print(f"ensemble={correct}/{total} ({correct / total:.6f})")
    print(
        f"majority_resolved={majority_resolved} fallback_used={fallback_used} "
        f"changed_from_fallback={changed_from_fallback}"
    )
    print(f"abstained_total={abstained_total}")
    print(f"output={args.output}")


def create_submission(args: argparse.Namespace) -> None:
    loaded = [load_submission_csv(path) for path in args.inputs]
    orders = [item[0] for item in loaded]
    mappings = [item[1] for item in loaded]
    order = validate_id_sets(args.inputs, orders, mappings)
    fallback_index = args.fallback_voter - 1
    if not 0 <= fallback_index < len(mappings):
        raise SystemExit("--fallback-voter is outside the input voter range")
    if len(order) != 831:
        raise SystemExit(f"Expected 831 submission rows, found {len(order)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    majority_resolved = 0
    fallback_used = 0
    changed_from_fallback = 0
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "answer"])
        for problem_id in order:
            votes = [mapping[problem_id] for mapping in mappings]
            answer, decision = choose_vote(votes, fallback_index)
            if answer is None:
                raise SystemExit(f"No integer ensemble answer for {problem_id}")
            majority_resolved += int(decision == "majority")
            fallback_used += int(decision == "fallback")
            changed_from_fallback += int(answer != votes[fallback_index])
            writer.writerow([problem_id, answer])

    print(f"samples={len(order)}")
    print(
        f"majority_resolved={majority_resolved} fallback_used={fallback_used} "
        f"changed_from_fallback={changed_from_fallback}"
    )
    print(f"submission={args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)

    evaluate_parser = commands.add_parser("evaluate")
    evaluate_parser.add_argument("inputs", nargs="+", type=Path)
    evaluate_parser.add_argument("--output", type=Path, required=True)
    evaluate_parser.add_argument("--fallback-voter", type=int, default=1)
    evaluate_parser.add_argument("--abstain-truncated", action="store_true")
    evaluate_parser.add_argument("--stuck-threshold", type=int, default=2046)
    evaluate_parser.set_defaults(function=evaluate)

    submission_parser = commands.add_parser("submission")
    submission_parser.add_argument("inputs", nargs="+", type=Path)
    submission_parser.add_argument("--output", type=Path, required=True)
    submission_parser.add_argument("--fallback-voter", type=int, default=1)
    submission_parser.set_defaults(function=create_submission)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
