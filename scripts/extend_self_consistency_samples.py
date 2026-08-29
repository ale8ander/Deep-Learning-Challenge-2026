import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ensemble_predictions import normalize_integer
from submit_baseline import SYSTEM_PROMPTS, extract_answer, response_token_count


DEFAULT_MODEL_PATH = Path("/workspace/models/Qwen2.5-3B-Instruct")


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    seen = set()
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = str(row.get("id", "")).strip()
            if not problem_id or problem_id in seen:
                raise SystemExit(f"Missing or duplicate id in {path}:{line_number}")
            if not isinstance(row.get("question"), str):
                raise SystemExit(f"Missing question in {path}:{line_number}")
            seen.add(problem_id)
            rows.append(row)
    return rows


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--min-baseline-support", type=int)
    parser.add_argument("--max-baseline-support", type=int)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--model-seed", type=int, default=20260901)
    parser.add_argument("--prompt-style", choices=tuple(SYSTEM_PROMPTS), default="default")
    args = parser.parse_args()

    rows = read_jsonl(args.source)
    if args.min_baseline_support is not None:
        rows = [
            row for row in rows
            if int(row.get("baseline_support", 0)) >= args.min_baseline_support
        ]
    if args.max_baseline_support is not None:
        rows = [
            row for row in rows
            if int(row.get("baseline_support", 0)) <= args.max_baseline_support
        ]
    if not rows:
        raise SystemExit("No rows selected")
    if args.num_samples < 2 or args.batch_size < 1:
        raise SystemExit("Invalid sampling configuration")

    torch.manual_seed(args.model_seed)
    torch.cuda.manual_seed_all(args.model_seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, local_files_only=True
    )
    model = PeftModel.from_pretrained(
        model, args.adapter_path, local_files_only=True
    ).cuda().eval()
    system_prompt = SYSTEM_PROMPTS[args.prompt_style]

    results = []
    started = time.time()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as output, torch.inference_mode():
        for start in tqdm(range(0, len(rows), args.batch_size), desc="extend-samples"):
            batch = rows[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": row["question"]},
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True).to(model.device)
            generated = model.generate(
                **encoded,
                do_sample=True,
                temperature=args.temperature,
                top_p=args.top_p,
                num_return_sequences=args.num_samples,
                max_new_tokens=args.max_new_tokens,
                pad_token_id=tokenizer.eos_token_id,
            )
            decoded = tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            for batch_index, source_row in enumerate(batch):
                responses = decoded[
                    batch_index * args.num_samples :
                    (batch_index + 1) * args.num_samples
                ]
                predictions = [
                    normalize_integer(extract_answer(response))
                    for response in responses
                ]
                result = {
                    "id": source_row["id"],
                    "question": source_row["question"],
                    "answer": normalize_integer(source_row.get("answer")),
                    "votes": source_row.get("votes"),
                    "baseline_prediction": normalize_integer(
                        source_row.get("baseline_prediction")
                    ),
                    "baseline_support": int(source_row.get("baseline_support", 0)),
                    "sample_predictions": predictions,
                    "sample_counts": dict(
                        Counter(value for value in predictions if value is not None).most_common()
                    ),
                    "valid_samples": sum(value is not None for value in predictions),
                    "responses": responses,
                    "response_tokens": [
                        response_token_count(tokenizer, response) for response in responses
                    ],
                    "source": str(args.source),
                    "model_seed": args.model_seed,
                    "prompt_style": args.prompt_style,
                }
                results.append(result)
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()

    elapsed = time.time() - started
    print(
        f"problems={len(results)} generations={len(results) * args.num_samples} "
        f"elapsed_seconds={elapsed:.1f}"
    )
    if all(row["answer"] is not None for row in results):
        sample_oracle = sum(
            row["answer"] in row["sample_predictions"] for row in results
        )
        print(f"sample_oracle={sample_oracle}/{len(results)}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
