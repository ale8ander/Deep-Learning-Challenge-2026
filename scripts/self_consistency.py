import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from baseline import DEFAULT_MODEL_PATH, SYSTEM_PROMPTS, extract_answer, validation_rows


def load_greedy_predictions(path: Path) -> tuple[list[str], dict[str, dict]]:
    order: list[str] = []
    rows: dict[str, dict] = {}
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
            if not isinstance(problem_id, str) or not problem_id:
                raise SystemExit(f"Missing id in {path}:{line_number}")
            if problem_id in rows:
                raise SystemExit(f"Duplicate id in {path}: {problem_id}")
            if not isinstance(row.get("response"), str):
                raise SystemExit(f"Missing response in {path}:{line_number}")
            order.append(problem_id)
            rows[problem_id] = row
    if not rows:
        raise SystemExit(f"No predictions found in {path}")
    return order, rows


def choose_prediction(
    greedy_prediction: str | None,
    sampled_predictions: list[str | None],
) -> tuple[str | None, str]:
    votes = [greedy_prediction, *sampled_predictions]
    counts = Counter(vote for vote in votes if vote is not None)
    if counts:
        winner, count = counts.most_common(1)[0]
        if count >= 2:
            return winner, "majority"
    return greedy_prediction, "greedy_fallback"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--greedy-predictions", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--adapter-scale", type=float, default=1.0)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--validation-ratio", type=float, default=0.1)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=1024)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--prompt-style", choices=SYSTEM_PROMPTS, default="default")
    parser.add_argument("--seed", type=int, default=2026)
    args = parser.parse_args()

    if args.adapter_scale < 0:
        raise SystemExit("--adapter-scale must be non-negative")
    if args.samples <= 0:
        raise SystemExit("--samples must be positive")
    if args.batch_size <= 0:
        raise SystemExit("--batch-size must be positive")
    if args.max_new_tokens <= 0:
        raise SystemExit("--max-new-tokens must be positive")
    if args.temperature <= 0:
        raise SystemExit("--temperature must be positive")
    if not 0 < args.top_p <= 1:
        raise SystemExit("--top-p must be in (0, 1]")

    greedy_order, greedy_rows = load_greedy_predictions(args.greedy_predictions)
    official_rows = validation_rows(args.data_dir, args.validation_ratio)
    official_by_id = {row["id"]: row for row in official_rows}
    missing = [problem_id for problem_id in greedy_order if problem_id not in official_by_id]
    if missing:
        raise SystemExit(f"Greedy IDs are not in the validation split: {missing[:10]}")
    rows = [official_by_id[problem_id] for problem_id in greedy_order]
    for row in rows:
        stored_answer = str(greedy_rows[row["id"]].get("answer", "")).strip()
        if stored_answer != row["answer"].strip():
            raise SystemExit(f"Official answer mismatch for {row['id']}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32
    torch.manual_seed(args.seed)
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed)

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=dtype,
        local_files_only=True,
    )
    from peft import PeftModel
    from peft.tuners.lora.layer import LoraLayer

    model = PeftModel.from_pretrained(model, args.adapter_path, local_files_only=True)
    for module in model.modules():
        if isinstance(module, LoraLayer):
            module.set_scale("default", args.adapter_scale)
    model = model.to(device).eval()

    system = SYSTEM_PROMPTS[args.prompt_style]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = 0
    greedy_correct = 0
    recovered = 0
    regressed = 0
    changed_but_still_wrong = 0
    majority_used = 0
    fallback_used = 0
    started = time.time()

    with args.output.open("w", encoding="utf-8") as output, torch.inference_mode():
        for start in tqdm(range(0, len(rows), args.batch_size), desc="self-consistency"):
            batch = rows[start : start + args.batch_size]
            conversations = [
                [
                    {"role": "system", "content": system},
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
            inputs = tokenizer(prompts, return_tensors="pt", padding=True).to(device)
            generated = model.generate(
                **inputs,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.samples,
                pad_token_id=tokenizer.eos_token_id,
            )
            decoded = tokenizer.batch_decode(
                generated[:, inputs["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )

            for batch_index, row in enumerate(batch):
                problem_id = row["id"]
                greedy_row = greedy_rows[problem_id]
                greedy_prediction = greedy_row.get("prediction")
                sampled_responses = decoded[
                    batch_index * args.samples : (batch_index + 1) * args.samples
                ]
                sampled_predictions = [
                    extract_answer(response) for response in sampled_responses
                ]
                prediction, decision = choose_prediction(
                    greedy_prediction, sampled_predictions
                )
                is_correct = prediction == row["answer"].strip()
                was_correct = greedy_prediction == row["answer"].strip()
                correct += int(is_correct)
                greedy_correct += int(was_correct)
                recovered += int(not was_correct and is_correct)
                regressed += int(was_correct and not is_correct)
                changed_but_still_wrong += int(
                    not was_correct and not is_correct and prediction != greedy_prediction
                )
                majority_used += int(decision == "majority")
                fallback_used += int(decision == "greedy_fallback")
                final_response = greedy_row["response"]
                if prediction != greedy_prediction:
                    for sampled_prediction, sampled_response in zip(
                        sampled_predictions, sampled_responses
                    ):
                        if sampled_prediction == prediction:
                            final_response = sampled_response
                            break
                output.write(
                    json.dumps(
                        {
                            "id": problem_id,
                            "answer": row["answer"].strip(),
                            "prediction": prediction,
                            "correct": is_correct,
                            "response": final_response,
                            "greedy_prediction": greedy_prediction,
                            "greedy_correct": was_correct,
                            "sampled_predictions": sampled_predictions,
                            "sampled_responses": sampled_responses,
                            "votes": [greedy_prediction, *sampled_predictions],
                            "decision": decision,
                            "temperature": args.temperature,
                            "top_p": args.top_p,
                            "seed": args.seed,
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )
                output.flush()

    total = len(rows)
    elapsed = time.time() - started
    print(f"device={device} samples={total}")
    print(f"greedy={greedy_correct}/{total} ({greedy_correct / total:.6f})")
    print(f"self_consistency={correct}/{total} ({correct / total:.6f})")
    print(
        f"recovered={recovered} regressed={regressed} "
        f"changed_but_still_wrong={changed_but_still_wrong}"
    )
    print(f"majority_used={majority_used} fallback_used={fallback_used}")
    print(f"elapsed_seconds={elapsed:.1f} output={args.output}")


if __name__ == "__main__":
    main()
