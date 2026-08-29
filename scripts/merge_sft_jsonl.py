import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path

from transformers import AutoTokenizer


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("inputs", type=Path, nargs="+")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/workspace/models/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--max-seq-length", type=int, default=2048)
    parser.add_argument(
        "--teacher-repeat",
        action="append",
        default=[],
        metavar="TEACHER=COUNT",
        help="Repeat rows from a teacher COUNT times in the training output.",
    )
    args = parser.parse_args()

    teacher_repeats = {}
    for value in args.teacher_repeat:
        teacher, separator, count_text = value.rpartition("=")
        if not separator or not teacher:
            raise SystemExit(f"Invalid --teacher-repeat value: {value}")
        try:
            count = int(count_text)
        except ValueError as error:
            raise SystemExit(f"Invalid repeat count: {value}") from error
        if count < 1:
            raise SystemExit(f"Repeat count must be at least 1: {value}")
        teacher_repeats[teacher] = count

    merged = {}
    duplicates = []
    source_counts = {}
    for path in args.inputs:
        rows = read_jsonl(path)
        source_counts[str(path)] = len(rows)
        for row in rows:
            problem_id = row["id"]
            if problem_id in merged:
                duplicates.append(problem_id)
                if row != merged[problem_id]:
                    raise SystemExit(f"Conflicting duplicate ID: {problem_id}")
            merged[problem_id] = row
    if duplicates:
        raise SystemExit(f"Duplicate IDs found: {duplicates[:10]}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    rows = [merged[key] for key in sorted(merged)]
    token_counts = []
    for row in rows:
        messages = row.get("messages")
        if not isinstance(messages, list) or len(messages) != 3:
            raise SystemExit(f"Invalid messages for {row['id']}")
        if [message.get("role") for message in messages] != ["system", "user", "assistant"]:
            raise SystemExit(f"Invalid message roles for {row['id']}")
        count = len(
            tokenizer.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=False,
            )
        )
        if count > args.max_seq_length:
            raise SystemExit(f"{row['id']} has {count} tokens (limit {args.max_seq_length})")
        token_counts.append(count)

    teachers = {row.get("teacher") for row in rows}
    unknown_teachers = set(teacher_repeats) - teachers
    if unknown_teachers:
        raise SystemExit(f"Unknown teachers in --teacher-repeat: {sorted(unknown_teachers)}")
    weighted_rows = []
    weighted_token_counts = []
    for row, token_count in zip(rows, token_counts):
        repeat = teacher_repeats.get(row.get("teacher"), 1)
        for repeat_index in range(1, repeat + 1):
            weighted = dict(row)
            if repeat > 1:
                weighted["training_repeat"] = repeat_index
            weighted_rows.append(weighted)
            weighted_token_counts.append(token_count)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in weighted_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest = {
        "inputs": [
            {"path": str(path), "sha256": sha256(path), "samples": source_counts[str(path)]}
            for path in args.inputs
        ],
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "unique_samples": len(rows),
        "samples": len(weighted_rows),
        "duplicates": len(duplicates),
        "teacher_repeats": teacher_repeats,
        "unique_teacher_counts": dict(Counter(row.get("teacher") for row in rows)),
        "weighted_teacher_counts": dict(
            Counter(row.get("teacher") for row in weighted_rows)
        ),
        "max_seq_length": args.max_seq_length,
        "minimum_tokens": min(weighted_token_counts),
        "average_tokens": sum(weighted_token_counts) / len(weighted_token_counts),
        "maximum_tokens": max(weighted_token_counts),
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
