"""Merge verbose teacher CoT into the hybrid SFT corpus.

The existing corpus pairs a "be concise" system prompt with ~222-character solutions
that assert correct steps without showing how they were found. Verbose regenerations
are paired instead with the neutral inference-time prompt, so the model learns to
produce the longer derivation under exactly the prompt evaluation uses.
"""

import argparse
import hashlib
import json
from pathlib import Path

from submit_baseline import SYSTEM_PROMPTS

VERBOSE_TEACHER_SUFFIX = "-verbose"


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


def retarget_system_prompt(row: dict) -> dict:
    """Pair the verbose solution with the prompt inference actually sends."""
    messages = []
    for message in row["messages"]:
        if message["role"] == "system":
            messages.append({"role": "system", "content": SYSTEM_PROMPTS["default"]})
        else:
            messages.append(message)
    out = dict(row)
    out["messages"] = messages
    teacher = row.get("teacher", "unknown")
    if not teacher.endswith(VERBOSE_TEACHER_SUFFIX):
        out["teacher"] = teacher + VERBOSE_TEACHER_SUFFIX
    return out


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base", type=Path, required=True)
    parser.add_argument("--verbose-sft", type=Path, action="append", required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    base = read_jsonl(args.base)
    base_by_id = {row["id"]: row for row in base}

    verbose: dict[str, dict] = {}
    for path in args.verbose_sft:
        for row in read_jsonl(path):
            if row["id"] in verbose:
                raise SystemExit(f"Duplicate verbose id across inputs: {row['id']}")
            verbose[row["id"]] = retarget_system_prompt(row)

    replaced = [i for i in verbose if i in base_by_id]
    added = [i for i in verbose if i not in base_by_id]

    merged = []
    for row in base:
        merged.append(verbose.get(row["id"], row))
    for problem_id in added:
        merged.append(verbose[problem_id])

    seen = set()
    for row in merged:
        if row["id"] in seen:
            raise SystemExit(f"Duplicate id in merged output: {row['id']}")
        seen.add(row["id"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in merged:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    def solution_length(row: dict) -> int:
        return len(
            next(m["content"] for m in row["messages"] if m["role"] == "assistant")
        )

    verbose_lengths = [solution_length(verbose[i]) for i in verbose]
    untouched = [r for r in merged if not r.get("teacher", "").endswith(VERBOSE_TEACHER_SUFFIX)]
    report = {
        "base": str(args.base),
        "base_samples": len(base),
        "verbose_inputs": [str(p) for p in args.verbose_sft],
        "verbose_samples": len(verbose),
        "replaced": len(replaced),
        "added": len(added),
        "output": str(args.output),
        "output_samples": len(merged),
        "output_sha256": sha256(args.output),
        "verbose_solution_chars_avg": round(sum(verbose_lengths) / len(verbose_lengths)),
        "untouched_samples": len(untouched),
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
