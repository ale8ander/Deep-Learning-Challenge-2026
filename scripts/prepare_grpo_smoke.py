import argparse
import csv
import hashlib
import json
from pathlib import Path


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def ids_from_jsonl(path: Path) -> set[str]:
    return {
        str(row["id"]).strip()
        for row in read_jsonl(path)
        if str(row.get("id", "")).startswith("train-")
    }


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-csv", type=Path, required=True)
    parser.add_argument("--screen-jsonl", type=Path, required=True)
    parser.add_argument("--exclude-csv", type=Path, action="append", default=[])
    parser.add_argument("--exclude-jsonl", type=Path, action="append", default=[])
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--size", type=int, default=128)
    parser.add_argument("--correct-fraction", type=float, default=0.25)
    parser.add_argument("--min-response-tokens", type=int, default=48)
    parser.add_argument("--max-response-tokens", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260825)
    args = parser.parse_args()

    if args.size <= 0:
        raise SystemExit("--size must be positive")
    if not 0 <= args.correct_fraction <= 1:
        raise SystemExit("--correct-fraction must be between 0 and 1")

    train_rows = read_csv(args.train_csv)
    excluded: set[str] = set()
    exclusion_counts: dict[str, int] = {}
    for path in args.exclude_csv:
        ids = {row["id"].strip() for row in read_csv(path)}
        exclusion_counts[str(path)] = len(ids)
        excluded.update(ids)
    for path in args.exclude_jsonl:
        ids = ids_from_jsonl(path)
        exclusion_counts[str(path)] = len(ids)
        excluded.update(ids)

    screen = {str(row["id"]).strip(): row for row in read_jsonl(args.screen_jsonl)}
    eligible: list[tuple[dict[str, str], dict]] = []
    for row in train_rows:
        problem_id = row["id"].strip()
        result = screen.get(problem_id)
        if problem_id in excluded or result is None or result.get("truncated"):
            continue
        response_tokens = int(
            result.get("response_tokens") or result.get("generated_tokens") or 0
        )
        if not args.min_response_tokens <= response_tokens <= args.max_response_tokens:
            continue
        eligible.append((row, result))

    def key(item: tuple[dict[str, str], dict]) -> str:
        return hashlib.sha256(f"{args.seed}:{item[0]['id'].strip()}".encode()).hexdigest()

    correct = sorted((item for item in eligible if item[1].get("correct")), key=key)
    wrong = sorted((item for item in eligible if not item[1].get("correct")), key=key)
    wanted_correct = round(args.size * args.correct_fraction)
    wanted_wrong = args.size - wanted_correct
    if len(correct) < wanted_correct or len(wrong) < wanted_wrong:
        raise SystemExit(
            f"Not enough eligible rows: correct {len(correct)}/{wanted_correct}, "
            f"wrong {len(wrong)}/{wanted_wrong}"
        )
    selected = correct[:wanted_correct] + wrong[:wanted_wrong]
    selected.sort(key=key)

    system_prompt = (
        "Solve the math problem carefully. The answer is always an integer. "
        "End your response with exactly: Final answer: <integer>"
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row, result in selected:
            record = {
                "id": row["id"].strip(),
                "prompt": [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": row["question"]},
                ],
                "answer": row["answer"].strip().replace(",", ""),
                "screen_correct": bool(result.get("correct")),
                "screen_prediction": result.get("prediction"),
                "screen_response_tokens": int(
                    result.get("response_tokens")
                    or result.get("generated_tokens")
                    or 0
                ),
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    selected_ids = {row["id"].strip() for row, _ in selected}
    manifest = {
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "samples": len(selected),
        "correct": sum(bool(result.get("correct")) for _, result in selected),
        "wrong": sum(not bool(result.get("correct")) for _, result in selected),
        "eligible_correct": len(correct),
        "eligible_wrong": len(wrong),
        "seed": args.seed,
        "response_token_range": [args.min_response_tokens, args.max_response_tokens],
        "excluded_unique_ids": len(excluded),
        "exclusion_counts": exclusion_counts,
        "selected_excluded_overlap": len(selected_ids & excluded),
        "selection": "stratified by greedy screen correctness, then sha256(seed:id)",
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
