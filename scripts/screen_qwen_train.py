import argparse
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import time
from typing import Optional

import torch
from tqdm import tqdm
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = Path("/workspace/models/Qwen2.5-3B-Instruct")
SYSTEM_PROMPT = (
    "Solve the math problem carefully. The answer is always an integer. "
    "End your response with exactly: Final answer: <integer>"
)
ANSWER_PATTERNS = (
    re.compile(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}"),
    re.compile(r"(?:final answer|answer|정답)\s*(?:is|:|=)?\s*(-?\d[\d,]*)", re.I),
    re.compile(r"(-?\d[\d,]*)"),
)
EXPECTED_COUNTS = {
    "official_total": 17000,
    "filtered_ids": 627,
    "clean": 16373,
    "screened_training": 14735,
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def is_validation_id(problem_id: str, ratio: float = 0.1) -> bool:
    value = int(hashlib.sha256(problem_id.encode()).hexdigest()[:8], 16) / 2**32
    return value < ratio


def official_training_rows(data_dir: Path) -> list[dict[str, str]]:
    rows = read_csv(data_dir / "deep_chal_math_train.csv")
    filtered_rows = read_csv(data_dir / "train_filtered_ids.csv")
    excluded = {row["id"] for row in filtered_rows}
    if len(excluded) != len(filtered_rows):
        raise SystemExit("Duplicate IDs found in train_filtered_ids.csv")
    clean = [row for row in rows if row["id"] not in excluded]
    training = [row for row in clean if not is_validation_id(row["id"])]
    counts = {
        "official_total": len(rows),
        "filtered_ids": len(excluded),
        "clean": len(clean),
        "screened_training": len(training),
    }
    if counts != EXPECTED_COUNTS:
        raise SystemExit(f"Unexpected official data counts: {counts}")
    return sorted(training, key=lambda row: row["id"])


def extract_answer(text: str) -> Optional[str]:
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].replace(",", "")
    return None


def read_existing(path: Path, expected_rows: dict[str, dict]) -> dict[str, dict]:
    if not path.exists():
        return {}
    if path.is_symlink() or not path.is_file():
        raise SystemExit(f"Refusing non-regular output path: {path}")
    existing = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as error:
                raise SystemExit(
                    f"Invalid JSON at {path}:{line_number}; do not resume this file"
                ) from error
            problem_id = row.get("id")
            if problem_id not in expected_rows:
                raise SystemExit(f"Unexpected ID in existing output: {problem_id}")
            if problem_id in existing:
                raise SystemExit(f"Duplicate ID in existing output: {problem_id}")
            if str(row.get("answer")) != expected_rows[problem_id]["answer"].strip():
                raise SystemExit(f"Answer mismatch in existing output: {problem_id}")
            existing[problem_id] = row
    return existing


def generated_token_info(token_ids: list[int], eos_token_id: int) -> tuple[int, bool]:
    try:
        eos_index = token_ids.index(eos_token_id)
    except ValueError:
        return len(token_ids), False
    return eos_index + 1, True


def atomic_write_jsonl(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    if temporary.exists():
        if temporary.is_symlink() or not temporary.is_file():
            raise SystemExit(f"Refusing unsafe temporary path: {temporary}")
        temporary.unlink()
    with temporary.open("w", encoding="utf-8") as file:
        for row in rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
        file.flush()
        os.fsync(file.fileno())
    os.replace(temporary, path)



def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/qwen_official_train_screen.jsonl"),
    )
    parser.add_argument(
        "--failures-output",
        type=Path,
        default=Path("data/processed/qwen_failure_candidates.jsonl"),
    )
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument(
        "--retry-truncated-from",
        type=Path,
        help="Only rerun IDs marked truncated in a previous screen JSONL.",
    )
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--limit", type=int, default=None, help="Smoke-test only.")
    args = parser.parse_args()

    if args.batch_size < 1 or args.max_new_tokens < 1:
        raise SystemExit("--batch-size and --max-new-tokens must be positive")
    if args.limit is not None and args.limit < 1:
        raise SystemExit("--limit must be positive")
    if not torch.cuda.is_available():
        raise SystemExit("CUDA GPU is required")

    all_rows = official_training_rows(args.data_dir)
    if args.retry_truncated_from is not None:
        previous_by_id = read_existing(
            args.retry_truncated_from,
            {row["id"]: row for row in all_rows},
        )
        truncated_ids = {
            problem_id
            for problem_id, result in previous_by_id.items()
            if result.get("truncated") is True
        }
        if not truncated_ids:
            raise SystemExit(
                f"No truncated rows found in {args.retry_truncated_from}"
            )
        all_rows = [row for row in all_rows if row["id"] in truncated_ids]
        print(
            f"retry_truncated_from={args.retry_truncated_from} "
            f"selected_truncated={len(all_rows)}",
            flush=True,
        )
    rows = all_rows if args.limit is None else all_rows[: args.limit]
    row_by_id = {row["id"]: row for row in rows}
    existing = read_existing(args.output, row_by_id)
    remaining = [row for row in rows if row["id"] not in existing]
    print(
        f"selected={len(rows)} resumed={len(existing)} remaining={len(remaining)}",
        flush=True,
    )

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16,
        local_files_only=True,
    ).to("cuda").eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    mode = "a" if args.output.exists() else "w"
    with args.output.open(mode, encoding="utf-8") as output, torch.inference_mode():
        for start in tqdm(
            range(0, len(remaining), args.batch_size),
            desc="screen official train",
        ):
            batch = remaining[start : start + args.batch_size]
            conversations = [
                [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                ]
                for row in batch
            ]
            prompts = [
                tokenizer.apply_chat_template(
                    messages,
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for messages in conversations
            ]
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to("cuda")
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            new_tokens = generated[:, inputs["input_ids"].shape[1] :]
            completions = tokenizer.batch_decode(new_tokens, skip_special_tokens=True)
            for row, completion, token_tensor in zip(batch, completions, new_tokens):
                token_ids = token_tensor.tolist()
                generated_tokens, finished = generated_token_info(
                    token_ids, tokenizer.eos_token_id
                )
                prediction = extract_answer(completion)
                answer = row["answer"].strip()
                result = {
                    "id": row["id"],
                    "answer": answer,
                    "prediction": prediction,
                    "correct": prediction == answer,
                    "truncated": not finished and generated_tokens >= args.max_new_tokens,
                    "generated_tokens": generated_tokens,
                    "response": completion,
                }
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
            output.flush()
            os.fsync(output.fileno())

    completed = read_existing(args.output, row_by_id)
    if len(completed) != len(rows) or set(completed) != set(row_by_id):
        raise SystemExit(
            f"Incomplete output: completed={len(completed)} expected={len(rows)}; "
            "Pod was not stopped"
        )
    ordered_results = [completed[row["id"]] for row in rows]
    failures = []
    for row, result in zip(rows, ordered_results):
        if result["correct"]:
            continue
        failures.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"].strip(),
                "qwen_prediction": result["prediction"],
                "qwen_truncated": result["truncated"],
                "qwen_generated_tokens": result["generated_tokens"],
                "qwen_response": result["response"],
            }
        )
    atomic_write_jsonl(args.failures_output, failures)

    elapsed = time.time() - started
    correct = len(rows) - len(failures)
    truncated = sum(bool(row["truncated"]) for row in ordered_results)
    manifest = {
        "base_model": "Qwen/Qwen2.5-3B-Instruct",
        "base_model_path": str(args.model_path),
        "data": str(args.data_dir / "deep_chal_math_train.csv"),
        "data_sha256": sha256(args.data_dir / "deep_chal_math_train.csv"),
        "filtered_ids": str(args.data_dir / "train_filtered_ids.csv"),
        "filtered_ids_sha256": sha256(args.data_dir / "train_filtered_ids.csv"),
        "selection": "clean official train excluding stable sha256 10% validation split",
        "samples": len(rows),
        "correct": correct,
        "accuracy": correct / len(rows),
        "failures": len(failures),
        "truncated": truncated,
        "resumed_rows": len(existing),
        "elapsed_seconds_current_run": elapsed,
        "batch_size": args.batch_size,
        "max_new_tokens": args.max_new_tokens,
        "seed": args.seed,
        "retry_truncated_from": (
            str(args.retry_truncated_from)
            if args.retry_truncated_from is not None
            else None
        ),
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "failures_output": str(args.failures_output),
        "failures_output_sha256": sha256(args.failures_output),
        "versions": {
            "torch": torch.__version__,
            "transformers": transformers.__version__,
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    with manifest_path.open("rb") as file:
        os.fsync(file.fileno())
    print(json.dumps(manifest, ensure_ascii=False, indent=2), flush=True)
    print("All outputs validated and flushed to disk.", flush=True)


if __name__ == "__main__":
    main()
