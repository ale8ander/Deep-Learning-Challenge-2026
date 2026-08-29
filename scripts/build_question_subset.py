import argparse
import csv
import json
from pathlib import Path


def read_reference_ids(path: Path) -> list[str]:
    ids = []
    seen = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = str(row.get("id", "")).strip()
            if not problem_id or problem_id in seen:
                raise SystemExit(f"Missing or duplicate id in {path}:{line_number}")
            seen.add(problem_id)
            ids.append(problem_id)
    return ids


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--reference-jsonl", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    with args.questions.open(encoding="utf-8-sig", newline="") as file:
        rows = list(csv.DictReader(file))
    required = {"id", "question"}
    if not rows or not required.issubset(rows[0]):
        raise SystemExit(f"Questions CSV must contain {sorted(required)}")
    by_id = {}
    for row in rows:
        problem_id = row["id"].strip()
        if not problem_id or problem_id in by_id:
            raise SystemExit(f"Missing or duplicate question id: {problem_id!r}")
        by_id[problem_id] = row

    reference_ids = read_reference_ids(args.reference_jsonl)
    missing = [problem_id for problem_id in reference_ids if problem_id not in by_id]
    if missing:
        raise SystemExit(f"Reference IDs missing from questions CSV: {missing[:10]}")

    fieldnames = [name for name in ("id", "question", "answer") if name in rows[0]]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=fieldnames)
        writer.writeheader()
        for problem_id in reference_ids:
            writer.writerow({name: by_id[problem_id][name] for name in fieldnames})
    print(f"rows={len(reference_ids)} output={args.output}")


if __name__ == "__main__":
    main()
