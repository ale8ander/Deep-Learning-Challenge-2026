import argparse
import csv
import json
import re
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


ANSWER_PATTERNS = (
    re.compile(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}"),
    re.compile(
        r"(?:final answer|answer|정답)\s*(?:is|:|=)?\s*(-?\d[\d,]*)",
        re.IGNORECASE,
    ),
    re.compile(r"(-?\d[\d,]*)"),
)

SYSTEM_PROMPTS = {
    "default": (
        "Solve the math problem carefully. "
        "The answer is always an integer. "
        "End your response with exactly: Final answer: <integer>"
    ),
    "verify": (
        "Solve the given math problem as a meticulous contest mathematician. "
        "First identify the exact quantity requested and all stated constraints. "
        "Work through the solution carefully. Before committing to the answer, "
        "independently verify it by substitution, enumeration, or a second method. "
        "Check signs, strict inequalities, integer ranges, ordered versus unordered "
        "counting, repeated cases, proper versus total divisors, overlapping regions, "
        "and whether a maximum or minimum was proved. Correct any failed check. "
        "Do not round unless explicitly required. The answer is always an integer. "
        "End your response with exactly: Final answer: <integer>"
    ),
    "rigor": (
        "Solve the math problem rigorously. Begin by listing the exact quantity requested "
        "and every condition that the final answer must satisfy. Maintain a checklist and "
        "explicitly verify that no condition is dropped. If the solution has multiple "
        "cases, branches, residue classes, configurations, or boundary cases, enumerate "
        "and finish all of them; never infer the general result from only some cases. "
        "After deriving a candidate integer answer, independently recompute or verify it "
        "by a genuinely separate method such as substitution, exhaustive enumeration, "
        "complementary counting, or checking all constraints. If verification reveals a "
        "contradiction or missed case, revise the reasoning and change the answer rather "
        "than reporting the original candidate. Do not guess when a derivation is "
        "incomplete. The answer is always an integer. End your response with exactly: "
        "Final answer: <integer>"
    ),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def extract_answer(text: str) -> str | None:
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
    max_new_tokens: int,
) -> list[str]:
    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
    ).to(model.device)
    generated = model.generate(
        **inputs,
        max_new_tokens=max_new_tokens,
        do_sample=False,
        pad_token_id=tokenizer.eos_token_id,
    )
    return tokenizer.batch_decode(
        generated[:, inputs["input_ids"].shape[1] :],
        skip_special_tokens=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/workspace/models/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data/deep_chal_math_leaderboard_filtered.csv"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/leaderboard_baseline.jsonl"),
    )
    parser.add_argument(
        "--submission",
        type=Path,
        default=Path("submission.csv"),
    )
    parser.add_argument("--expected-rows", type=int, default=831)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--retry-max-new-tokens", type=int)
    parser.add_argument("--retry-batch-size", type=int, default=4)
    parser.add_argument("--temperature", type=float)
    parser.add_argument("--top-p", type=float, default=1.0)
    parser.add_argument("--model-seed", type=int, default=20260828)
    parser.add_argument(
        "--prompt-style",
        choices=tuple(SYSTEM_PROMPTS),
        default="default",
    )
    args = parser.parse_args()

    if args.expected_rows <= 0:
        raise SystemExit("--expected-rows must be positive")
    if args.adapter_scale < 0:
        raise SystemExit("--adapter-scale must be non-negative")
    if args.retry_batch_size <= 0:
        raise SystemExit("--retry-batch-size must be positive")
    if args.temperature is not None and args.temperature <= 0:
        raise SystemExit("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise SystemExit("--top-p must be in (0, 1]")
    if (
        args.retry_max_new_tokens is not None
        and args.retry_max_new_tokens <= args.max_new_tokens
    ):
        raise SystemExit("--retry-max-new-tokens must exceed --max-new-tokens")

    rows = read_csv(args.input)
    if len(rows) != args.expected_rows:
        raise SystemExit(
            f"Expected {args.expected_rows} rows, found {len(rows)}"
        )

    ids = [row["id"].strip() for row in rows]
    if len(ids) != len(set(ids)):
        raise SystemExit("Duplicate IDs found in leaderboard data")

    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path,
        local_files_only=True,
    )
    tokenizer.padding_side = "left"

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16,
        local_files_only=True,
    )
    if args.adapter_path:
        from peft import PeftModel
        from peft.tuners.lora.layer import LoraLayer

        model = PeftModel.from_pretrained(
            model,
            args.adapter_path,
            local_files_only=True,
        )
        for module in model.modules():
            if isinstance(module, LoraLayer):
                module.set_scale("default", args.adapter_scale)
    model = model.cuda().eval()

    model.generation_config.do_sample = args.temperature is not None
    model.generation_config.temperature = args.temperature
    model.generation_config.top_p = (
        args.top_p if args.temperature is not None else None
    )
    model.generation_config.top_k = None

    system_prompt = SYSTEM_PROMPTS[args.prompt_style]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.submission.parent.mkdir(parents=True, exist_ok=True)
    predictions: list[tuple[str, str | None]] = []
    has_ground_truth = "answer" in rows[0]
    correct_predictions = 0
    retried_truncated = 0
    started = time.time()

    with (
        args.output.open("w", encoding="utf-8") as output_file,
        torch.inference_mode(),
    ):
        for start in tqdm(
            range(0, len(rows), args.batch_size),
            desc="leaderboard",
        ):
            batch = rows[start : start + args.batch_size]
            conversations = [
                [
                    {"role": "system", "content": system_prompt},
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
            initial_completions = generate_responses(
                model, tokenizer, prompts, args.max_new_tokens
            )
            completions = list(initial_completions)
            initial_token_counts = [
                response_token_count(tokenizer, response)
                for response in initial_completions
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
                        args.retry_max_new_tokens,
                    )
                    for index, retry_completion in zip(indices, retry_completions):
                        completions[index] = retry_completion
                retried_truncated += len(retry_indices)
            retry_index_set = set(retry_indices)

            for index, (row, response) in enumerate(zip(batch, completions)):
                initial_response = initial_completions[index]
                prediction = extract_answer(response)
                predictions.append((row["id"].strip(), prediction))
                result = {
                    "id": row["id"].strip(),
                    "prediction": prediction,
                    "response": response,
                    "response_tokens": response_token_count(tokenizer, response),
                    "retried_truncated": index in retry_index_set,
                }
                if has_ground_truth:
                    answer = row["answer"].strip()
                    correct = prediction == answer
                    result.update({"answer": answer, "correct": correct})
                    correct_predictions += int(correct)
                if index in retry_index_set:
                    result.update(
                        {
                            "initial_prediction": extract_answer(initial_response),
                            "initial_response": initial_response,
                            "initial_response_tokens": initial_token_counts[index],
                        }
                    )
                output_file.write(
                    json.dumps(result, ensure_ascii=False) + "\n"
                )
                output_file.flush()

    missing = [problem_id for problem_id, answer in predictions if answer is None]
    if missing:
        raise SystemExit(
            f"Could not extract integer answers for {len(missing)} rows: "
            + ", ".join(missing)
            + f"\nInspect {args.output}; submission was not created."
        )

    with args.submission.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(["id", "answer"])
        writer.writerows(predictions)

    elapsed = time.time() - started
    print(f"samples={len(predictions)} elapsed_seconds={elapsed:.1f}")
    if has_ground_truth:
        print(
            f"accuracy={correct_predictions}/{len(predictions)} "
            f"({correct_predictions / len(predictions):.5f})"
        )
    print(f"retried_truncated={retried_truncated}")
    print(f"details={args.output}")
    print(f"submission={args.submission}")


if __name__ == "__main__":
    main()
