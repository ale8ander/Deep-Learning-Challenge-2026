import argparse
import csv
import hashlib
import json
import re
import time
from pathlib import Path
from typing import Optional

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = Path("/workspace/models/Qwen2.5-3B-Instruct")
SYSTEM_PROMPTS = {
    "default": (
        "Solve the math problem carefully. The answer is always an integer. "
        "End your response with exactly: Final answer: <integer>"
    ),
    "verify": ("Solve the given math problem as a meticulous contest mathematician. First identify the exact quantity requested and all stated constraints. Work through the solution carefully. Before committing to the answer, independently verify it by substitution, enumeration, or a second method. Check signs, strict inequalities, integer ranges, ordered versus unordered counting, repeated cases, proper versus total divisors, overlapping regions, and whether a maximum or minimum was proved. Correct any failed check. Do not round unless explicitly required. The answer is always an integer. End your response with exactly: Final answer: <integer>"),
    "concise": (
        "Solve the math problem accurately and keep the reasoning concise. "
        "The answer is always an integer. Use at most 400 tokens. "
        "End your response with exactly: Final answer: <integer>"
    ),
}
ANSWER_PATTERNS = (
    re.compile(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}"),
    re.compile(r"(?:final answer|answer|정답)\s*(?:is|:|=)?\s*(-?\d[\d,]*)", re.I),
    re.compile(r"(-?\d[\d,]*)"),
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def validation_rows(data_dir: Path, ratio: float) -> list[dict[str, str]]:
    rows = read_csv(data_dir / "deep_chal_math_train.csv")
    excluded = {row["id"] for row in read_csv(data_dir / "train_filtered_ids.csv")}
    clean = [row for row in rows if row["id"] not in excluded]

    # Stable across machines and independent of input row order.
    selected = [
        row
        for row in clean
        if int(hashlib.sha256(row["id"].encode()).hexdigest()[:8], 16) / 2**32 < ratio
    ]
    return sorted(selected, key=lambda row: row["id"])


def extract_answer(text: str) -> Optional[str]:
    for pattern in ANSWER_PATTERNS:
        matches = pattern.findall(text)
        if matches:
            return matches[-1].replace(",", "")
    return None


def response_token_count(tokenizer: AutoTokenizer, response: str) -> int:
    return len(tokenizer(response, add_special_tokens=False)["input_ids"])


def generate_responses(
    model: AutoModelForCausalLM,
    tokenizer: AutoTokenizer,
    prompts: list[str],
    device: str,
    max_new_tokens: int,
) -> list[str]:
    inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(
        generated[:, inputs["input_ids"].shape[1] :], skip_special_tokens=True
    )


def truncated_ids_from_predictions(
    path: Path, tokenizer: AutoTokenizer, token_limit: int
) -> set[str]:
    if token_limit <= 2:
        raise SystemExit("--retry-token-limit must be greater than 2")

    seen: set[str] = set()
    selected: set[str] = set()
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
            response = row.get("response")
            if not isinstance(problem_id, str) or not problem_id:
                raise SystemExit(f"Missing id in {path}:{line_number}")
            if problem_id in seen:
                raise SystemExit(f"Duplicate id in {path}: {problem_id}")
            if not isinstance(response, str):
                raise SystemExit(f"Missing response in {path}:{line_number}")
            seen.add(problem_id)

            token_count = response_token_count(tokenizer, response)
            if token_count >= token_limit - 2:
                selected.add(problem_id)

    if not seen:
        raise SystemExit(f"No predictions found in {path}")
    if not selected:
        raise SystemExit(
            f"No responses reached the {token_limit}-token retry threshold in {path}"
        )
    print(
        f"retry_source={path} source_samples={len(seen)} "
        f"selected_truncated={len(selected)} retry_token_limit={token_limit}"
    )
    return selected


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--input",
        type=Path,
        help="Evaluate every row in this id,question,answer CSV instead of a hash split.",
    )
    parser.add_argument("--output", type=Path, default=Path("outputs/baseline.jsonl"))
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-max-new-tokens", type=int)
    parser.add_argument("--retry-batch-size", type=int, default=4)
    parser.add_argument("--retry-truncated-from", type=Path)
    parser.add_argument("--retry-token-limit", type=int, default=1024)
    parser.add_argument("--prompt-style", choices=SYSTEM_PROMPTS, default="default")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.adapter_scale < 0:
        raise SystemExit("--adapter-scale must be non-negative")
    if args.retry_batch_size <= 0:
        raise SystemExit("--retry-batch-size must be positive")
    if (
        args.retry_max_new_tokens is not None
        and args.retry_max_new_tokens <= args.max_new_tokens
    ):
        raise SystemExit("--retry-max-new-tokens must exceed --max-new-tokens")

    rows = (
        read_csv(args.input)
        if args.input is not None
        else validation_rows(args.data_dir, args.validation_ratio)
    )

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if args.retry_truncated_from is not None:
        selected_ids = truncated_ids_from_predictions(
            args.retry_truncated_from, tokenizer, args.retry_token_limit
        )
        validation_ids = {row["id"] for row in rows}
        missing_ids = selected_ids - validation_ids
        if missing_ids:
            examples = ",".join(sorted(missing_ids)[:10])
            raise SystemExit(
                f"Retry IDs are not in the validation split: {examples}"
            )
        rows = [row for row in rows if row["id"] in selected_ids]
    if args.limit is not None:
        rows = rows[: args.limit]
    if not rows:
        raise SystemExit("No validation rows selected.")

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device in {"mps", "cuda"} else torch.float32
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=dtype, local_files_only=True
    )
    if args.adapter_path:
        from peft import PeftModel
        from peft.tuners.lora.layer import LoraLayer

        model = PeftModel.from_pretrained(model, args.adapter_path, local_files_only=True)
        for module in model.modules():
            if isinstance(module, LoraLayer):
                module.set_scale("default", args.adapter_scale)
    model = model.to(device).eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    system = SYSTEM_PROMPTS[args.prompt_style]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    retried_truncated = 0
    retry_recovered = 0
    retry_regressed = 0
    started = time.time()

    with args.output.open("w", encoding="utf-8") as output, torch.inference_mode():
        for start in tqdm(range(0, len(rows), args.batch_size), desc="baseline"):
            batch = rows[start : start + args.batch_size]
            conversations = [
                [{"role": "system", "content": system}, {"role": "user", "content": row["question"]}]
                for row in batch
            ]
            prompts = [
                tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
                for messages in conversations
            ]
            initial_completions = generate_responses(
                model, tokenizer, prompts, device, args.max_new_tokens
            )
            completions = list(initial_completions)
            initial_token_counts = [
                response_token_count(tokenizer, completion)
                for completion in initial_completions
            ]
            retry_indices: list[int] = []
            if args.retry_max_new_tokens is not None:
                retry_indices = [
                    index
                    for index, token_count in enumerate(initial_token_counts)
                    if token_count >= args.max_new_tokens - 2
                ]
                for retry_start in range(0, len(retry_indices), args.retry_batch_size):
                    indices = retry_indices[
                        retry_start : retry_start + args.retry_batch_size
                    ]
                    retry_completions = generate_responses(
                        model,
                        tokenizer,
                        [prompts[index] for index in indices],
                        device,
                        args.retry_max_new_tokens,
                    )
                    for index, retry_completion in zip(indices, retry_completions):
                        completions[index] = retry_completion
                retried_truncated += len(retry_indices)
            retry_index_set = set(retry_indices)

            for index, (row, completion) in enumerate(zip(batch, completions)):
                initial_completion = initial_completions[index]
                initial_prediction = extract_answer(initial_completion)
                prediction = extract_answer(completion)
                is_correct = prediction == row["answer"].strip()
                initial_correct = initial_prediction == row["answer"].strip()
                if index in retry_index_set:
                    retry_recovered += int(not initial_correct and is_correct)
                    retry_regressed += int(initial_correct and not is_correct)
                correct += is_correct
                result = {
                    "id": row["id"],
                    "answer": row["answer"].strip(),
                    "prediction": prediction,
                    "correct": is_correct,
                    "response": completion,
                    "response_tokens": response_token_count(tokenizer, completion),
                    "retried_truncated": index in retry_index_set,
                }
                if index in retry_index_set:
                    result.update(
                        {
                            "initial_prediction": initial_prediction,
                            "initial_correct": initial_correct,
                            "initial_response": initial_completion,
                            "initial_response_tokens": initial_token_counts[index],
                        }
                    )
                output.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                output.flush()

    elapsed = time.time() - started
    print(f"device={device} samples={len(rows)} correct={correct}")
    print(f"accuracy={correct / len(rows):.6f} elapsed_seconds={elapsed:.1f}")
    print(
        f"retried_truncated={retried_truncated} "
        f"retry_recovered={retry_recovered} retry_regressed={retry_regressed}"
    )
    print(f"predictions={args.output}")


if __name__ == "__main__":
    main()
