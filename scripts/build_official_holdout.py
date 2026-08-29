import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def jsonl_ids(path: Path) -> set[str]:
    ids = set()
    with path.open(encoding="utf-8") as file:
        for line in file:
            if line.strip():
                row = json.loads(line)
                problem_id = row.get("id")
                if isinstance(problem_id, str) and problem_id.startswith("train-"):
                    ids.add(problem_id)
    return ids


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--filtered-ids", type=Path, required=True)
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[])
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--exclude-eval-jsonl", type=Path)
    parser.add_argument("--screen-jsonl", type=Path)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=1000)
    parser.add_argument("--seed", type=int, default=20260824)
    args = parser.parse_args()

    rows = read_csv(args.train_csv)
    excluded = {row["id"].strip() for row in read_csv(args.filtered_ids)}
    exclusion_counts = {"filtered_ids": len(excluded)}
    for path in args.exclude_csv:
        ids = {row["id"].strip() for row in read_csv(path)}
        exclusion_counts[str(path)] = len(ids)
        excluded.update(ids)
    for path in args.exclude_jsonl:
        ids = jsonl_ids(path)
        exclusion_counts[str(path)] = len(ids)
        excluded.update(ids)
    if args.exclude_eval_jsonl:
        ids = jsonl_ids(args.exclude_eval_jsonl)
        exclusion_counts[str(args.exclude_eval_jsonl)] = len(ids)
        excluded.update(ids)

    eligible = [row for row in rows if row["id"].strip() not in excluded]

    def selection_key(row: dict[str, str]) -> str:
        problem_id = row["id"].strip()
        return hashlib.sha256(f"{args.seed}:{problem_id}".encode()).hexdigest()

    eligible.sort(key=selection_key)
    if len(eligible) < args.size:
        raise SystemExit(f"Not enough eligible rows: {len(eligible)} < {args.size}")
    selected = eligible[: args.size]
    selected.sort(key=lambda row: row["id"].strip())

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=["id", "question", "answer"])
        writer.writeheader()
        writer.writerows(
            {
                "id": row["id"].strip(),
                "question": row["question"],
                "answer": row["answer"].strip(),
            }
            for row in selected
        )

    screen_distribution = Counter()
    if args.screen_jsonl:
        screen = {}
        with args.screen_jsonl.open(encoding="utf-8") as file:
            for line in file:
                if line.strip():
                    row = json.loads(line)
                    screen[row["id"]] = row
        for row in selected:
            result = screen.get(row["id"])
            if result is None:
                screen_distribution["missing"] += 1
            elif result.get("truncated"):
                screen_distribution["truncated"] += 1
            elif result.get("correct"):
                screen_distribution["correct"] += 1
            else:
                screen_distribution["wrong"] += 1

    selected_ids = {row["id"].strip() for row in selected}
    overlap = selected_ids & excluded
    manifest = {
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "samples": len(selected),
        "seed": args.seed,
        "source_rows": len(rows),
        "eligible_rows": len(eligible),
        "excluded_unique_ids": len(excluded),
        "exclusion_counts": exclusion_counts,
        "selected_excluded_overlap": len(overlap),
        "qwen_screen_distribution": dict(sorted(screen_distribution.items())),
        "selection": "lowest sha256(seed:id) among eligible rows",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
