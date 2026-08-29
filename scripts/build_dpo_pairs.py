import argparse
import hashlib
import json
from pathlib import Path


def load_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chosen-data", type=Path, required=True)
    parser.add_argument("--screen-data", type=Path, required=True)
    parser.add_argument("--train-output", type=Path, required=True)
    parser.add_argument("--dev-output", type=Path, required=True)
    parser.add_argument("--dev-size", type=int, default=100)
    args = parser.parse_args()

    chosen_rows = {
        row["id"]: row
        for row in load_jsonl(args.chosen_data)
        if str(row.get("id", "")).startswith("train-")
    }
    screen_rows = {row["id"]: row for row in load_jsonl(args.screen_data)}
    pairs = []
    for problem_id in sorted(set(chosen_rows) & set(screen_rows)):
        chosen_row = chosen_rows[problem_id]
        screen_row = screen_rows[problem_id]
        if screen_row.get("correct") or screen_row.get("truncated"):
            continue
        rejected = screen_row.get("response")
        messages = chosen_row.get("messages")
        if not isinstance(rejected, str) or not rejected.strip():
            continue
        if not isinstance(messages, list) or len(messages) < 2:
            continue
        chosen = messages[-1].get("content")
        if not isinstance(chosen, str) or not chosen.strip():
            continue
        pairs.append(
            {
                "id": problem_id,
                "question": chosen_row["question"],
                "answer": str(chosen_row["answer"]),
                "chosen": chosen,
                "rejected": rejected,
                "chosen_teacher": chosen_row.get("teacher"),
                "rejected_prediction": screen_row.get("prediction"),
            }
        )

    pairs.sort(
        key=lambda row: hashlib.sha256(row["id"].encode()).hexdigest()
    )
    if args.dev_size <= 0 or args.dev_size >= len(pairs):
        raise SystemExit("--dev-size must be between 1 and pair_count - 1")
    dev = pairs[: args.dev_size]
    train = pairs[args.dev_size :]
    for path, rows in ((args.train_output, train), (args.dev_output, dev)):
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            for row in rows:
                file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"pairs={len(pairs)} train={len(train)} dev={len(dev)}")
    print(f"train_output={args.train_output}")
    print(f"dev_output={args.dev_output}")


if __name__ == "__main__":
    main()
