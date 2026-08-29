import argparse
import csv
import hashlib
import json
from pathlib import Path


SYSTEM_PROMPT = (
    "You are a rigorous math-solution judge. Compare Response A and Response B "
    "against the problem, checking the reasoning and final integer. Choose the "
    "response that is mathematically correct. Reply with exactly A or B."
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
            if not isinstance(row.get("response"), str):
                raise SystemExit(f"Missing response in {path}:{line_number}")
            rows[problem_id] = row
    return rows


def normalize(value) -> str:
    return str(value).strip().replace(",", "")


def shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 3 // 4
    tail = max_chars - head
    return text[:head] + "\n[...middle omitted...]\n" + text[-tail:]


def split_is_dev(problem_id: str, seed: int, fraction: float) -> bool:
    value = int(hashlib.sha256(f"{seed}:{problem_id}".encode()).hexdigest()[:16], 16)
    return value / 2**64 < fraction


def pair_key(problem_id: str, left: int, right: int, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{problem_id}:{left}:{right}".encode()).hexdigest()


def make_record(
    problem_id: str,
    question: str,
    response_a: str,
    response_b: str,
    label: str,
    suffix: str,
) -> dict:
    user = (
        f"Problem:\n{question}\n\n"
        f"Response A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        "Which response is correct? Reply exactly A or B."
    )
    return {
        "id": f"selector:{problem_id}:{suffix}",
        "problem_id": problem_id,
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
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--dev-output", type=Path, required=True)
    parser.add_argument("--dev-fraction", type=float, default=0.15)
    parser.add_argument("--max-pairs-per-problem", type=int, default=6)
    parser.add_argument("--max-response-chars", type=int, default=5000)
    # Keep this distinct from the selector-pool selection seed; reusing that
    # seed would put the lowest-hash selected pool disproportionately in dev.
    parser.add_argument("--seed", type=int, default=20260827)
    args = parser.parse_args()

    if len(args.predictions) < 2:
        raise SystemExit("At least two --predictions files are required")
    if not 0 < args.dev_fraction < 1:
        raise SystemExit("--dev-fraction must be between 0 and 1")
    questions = read_csv(args.questions)
    candidates = [read_jsonl(path) for path in args.predictions]
    question_ids = [row["id"].strip() for row in questions]
    expected = set(question_ids)
    for path, mapping in zip(args.predictions, candidates, strict=True):
        if set(mapping) != expected:
            raise SystemExit(f"ID set mismatch in {path}")

    train_rows: list[dict] = []
    dev_rows: list[dict] = []
    useful_problems = 0
    skipped_no_contrast = 0
    for question_row in questions:
        problem_id = question_row["id"].strip()
        answer = normalize(question_row["answer"])
        correct = []
        wrong = []
        seen = set()
        for index, mapping in enumerate(candidates):
            row = mapping[problem_id]
            prediction = normalize(row.get("prediction"))
            response = shorten(row["response"], args.max_response_chars)
            signature = (prediction, response)
            if signature in seen:
                continue
            seen.add(signature)
            item = (index, response)
            (correct if prediction == answer else wrong).append(item)
        pairs = [
            (good, bad)
            for good in correct
            for bad in wrong
        ]
        pairs.sort(key=lambda pair: pair_key(problem_id, pair[0][0], pair[1][0], args.seed))
        pairs = pairs[: args.max_pairs_per_problem]
        if not pairs:
            skipped_no_contrast += 1
            continue
        useful_problems += 1
        target = dev_rows if split_is_dev(problem_id, args.seed, args.dev_fraction) else train_rows
        for pair_index, (good, bad) in enumerate(pairs):
            target.append(
                make_record(
                    problem_id,
                    question_row["question"],
                    good[1],
                    bad[1],
                    "A",
                    f"{pair_index}:ab",
                )
            )
            target.append(
                make_record(
                    problem_id,
                    question_row["question"],
                    bad[1],
                    good[1],
                    "B",
                    f"{pair_index}:ba",
                )
            )

    write_jsonl(args.train_output, train_rows)
    write_jsonl(args.dev_output, dev_rows)
    manifest = {
        "questions": str(args.questions),
        "prediction_files": [str(path) for path in args.predictions],
        "source_problems": len(questions),
        "useful_problems": useful_problems,
        "skipped_no_correct_vs_wrong_contrast": skipped_no_contrast,
        "train_pairs": len(train_rows),
        "dev_pairs": len(dev_rows),
        "train_problems": len({row["problem_id"] for row in train_rows}),
        "dev_problems": len({row["problem_id"] for row in dev_rows}),
        "problem_overlap": len(
            {row["problem_id"] for row in train_rows}
            & {row["problem_id"] for row in dev_rows}
        ),
        "a_labels": sum(row["messages"][-1]["content"] == "A" for row in train_rows + dev_rows),
        "b_labels": sum(row["messages"][-1]["content"] == "B" for row in train_rows + dev_rows),
        "max_pairs_per_problem_before_ab_swap": args.max_pairs_per_problem,
        "max_response_chars": args.max_response_chars,
        "seed": args.seed,
        "train_output": str(args.train_output),
        "train_sha256": sha256(args.train_output),
        "dev_output": str(args.dev_output),
        "dev_sha256": sha256(args.dev_output),
    }
    manifest_path = args.train_output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
