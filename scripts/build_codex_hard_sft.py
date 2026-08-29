import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re


SYSTEM_PROMPT = (
    "Solve the math problem independently. Give a concise, logically complete derivation. "
    "Do not restate the problem or explore multiple approaches. The answer is always an "
    "integer. End with exactly: Final answer: <integer>"
)
INTEGER_PATTERN = re.compile(r"[+-]?\d+")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--solutions", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    source_rows = read_jsonl(args.source)
    solution_rows = read_jsonl(args.solutions)
    source_by_id = {row["id"]: row for row in source_rows}
    if len(source_by_id) != len(source_rows):
        raise SystemExit("Duplicate IDs in source")

    selected = []
    skipped = []
    for solved in solution_rows:
        status = solved["status"]
        answer = solved.get("final_answer")
        if status not in {"ACCEPTED", "LABEL_ERROR"}:
            skipped.append({"id": solved["id"], "reason": status})
            continue
        answer_text = str(answer) if answer is not None else ""
        if INTEGER_PATTERN.fullmatch(answer_text) is None:
            skipped.append({"id": solved["id"], "reason": "non_integer_or_missing_answer"})
            continue
        source = source_by_id.get(solved["id"])
        if source is None:
            raise SystemExit(f"Missing source row: {solved['id']}")
        solution = solved["solution"].strip()
        solution = re.sub(r"\n*Final answer:\s*[+-]?\d+\s*$", "", solution).rstrip()
        selected.append(
            {
                "id": solved["id"],
                "question": source["question"].strip(),
                "answer": answer_text,
                "teacher": "codex-gpt-5.6-sol",
                "metadata": {
                    "source": str(args.source),
                    "review_status": status,
                    "original_answer": str(solved["official_answer"]),
                    "label_corrected": status == "LABEL_ERROR",
                },
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": source["question"].strip()},
                    {
                        "role": "assistant",
                        "content": f"{solution}\n\nFinal answer: {answer_text}",
                    },
                ],
            }
        )

    selected.sort(key=lambda row: row["id"])
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in selected:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "source": str(args.source),
        "source_sha256": sha256(args.source),
        "solutions": str(args.solutions),
        "solutions_sha256": sha256(args.solutions),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "samples": len(selected),
        "status_counts": dict(Counter(row["metadata"]["review_status"] for row in selected)),
        "skipped_counts": dict(Counter(row["reason"] for row in skipped)),
        "skipped": skipped,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
