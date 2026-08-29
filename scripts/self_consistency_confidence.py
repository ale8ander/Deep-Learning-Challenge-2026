import argparse
import csv
import hashlib
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


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_predictions(path: Path) -> dict[str, dict]:
    if path.suffix.lower() == ".csv":
        rows = {}
        for line_number, row in enumerate(read_csv(path), start=2):
            problem_id = str(row.get("id", "")).strip()
            prediction = row.get("prediction", row.get("answer"))
            if not problem_id or problem_id in rows:
                raise SystemExit(f"Missing or duplicate id in {path}:{line_number}")
            rows[problem_id] = {"id": problem_id, "prediction": prediction}
        return rows

    rows = {}
    with path.open(encoding="utf-8") as file:
        for line_number, line in enumerate(file, start=1):
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = str(row.get("id", "")).strip()
            if not problem_id or problem_id in rows:
                raise SystemExit(f"Missing or duplicate id in {path}:{line_number}")
            rows[problem_id] = row
    return rows


def stable_key(problem_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{problem_id}".encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-samples", type=int, default=8)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.95)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument(
        "--prompt-style", choices=tuple(SYSTEM_PROMPTS), default="default"
    )
    args = parser.parse_args()

    if args.num_samples <= 1 or args.batch_size <= 0:
        raise SystemExit("num-samples must exceed 1; batch-size must be positive")
    if not 0 < args.temperature or not 0 < args.top_p <= 1:
        raise SystemExit("Invalid temperature or top-p")

    questions = read_csv(args.questions)
    if not questions or not {"id", "question"}.issubset(questions[0]):
        raise SystemExit("Questions CSV must contain id and question")
    has_ground_truth = "answer" in questions[0]
    baseline_map = read_predictions(args.predictions)
    question_ids = [row["id"].strip() for row in questions]
    if set(baseline_map) != set(question_ids):
        raise SystemExit("--predictions id set does not match --questions id set")

    prepared = []
    for row in questions:
        problem_id = row["id"].strip()
        answer = normalize_integer(row["answer"]) if has_ground_truth else None
        baseline_prediction = normalize_integer(baseline_map[problem_id].get("prediction"))
        prepared.append(
            {
                "id": problem_id,
                "question": row["question"],
                "answer": answer,
                "baseline_prediction": baseline_prediction,
            }
        )
    prepared.sort(key=lambda row: stable_key(row["id"], args.seed))

    torch.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
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
        for start in tqdm(range(0, len(prepared), args.batch_size), desc="confidence"):
            batch = prepared[start : start + args.batch_size]
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
                output_scores=True,
                return_dict_in_generate=True,
            )
            sequences = generated.sequences
            input_len = encoded["input_ids"].shape[1]
            gen_tokens = sequences[:, input_len:]
            # Computed one sequence at a time: compute_transition_scores does an
            # internal log_softmax over the full (batch*num_samples, steps, vocab)
            # tensor when called on the whole batch at once, which OOMs at
            # batch_size*num_samples ~= 32 with a ~150k vocab. Per-sequence calls
            # keep the transient log_softmax buffer to a single sequence's size.
            num_sequences = sequences.shape[0]
            all_avg_logprobs: list[float | None] = []
            for sample_index in range(num_sequences):
                single_scores = tuple(step[sample_index : sample_index + 1] for step in generated.scores)
                single_seq = sequences[sample_index : sample_index + 1]
                transition = model.compute_transition_scores(
                    single_seq, single_scores, normalize_logits=True
                )[0]
                valid = transition[transition > -1e8]
                all_avg_logprobs.append(
                    float(valid.mean().item()) if valid.numel() > 0 else None
                )
                del single_scores, transition, valid
            decoded = tokenizer.batch_decode(gen_tokens, skip_special_tokens=True)

            for batch_index, row in enumerate(batch):
                lo = batch_index * args.num_samples
                hi = lo + args.num_samples
                responses = decoded[lo:hi]
                predictions = [normalize_integer(extract_answer(response)) for response in responses]
                avg_logprobs = all_avg_logprobs[lo:hi]

                valid_predictions = [p for p in predictions if p is not None]
                counts = Counter(valid_predictions)
                answer = row["answer"]
                result = {
                    "id": row["id"],
                    "answer": answer,
                    "baseline_prediction": row["baseline_prediction"],
                    "baseline_correct": (
                        row["baseline_prediction"] == answer if has_ground_truth else None
                    ),
                    "sample_predictions": predictions,
                    "sample_avg_logprob": avg_logprobs,
                    "sample_counts": dict(counts.most_common()),
                    "valid_samples": len(valid_predictions),
                    "any_sample_correct": (
                        answer in valid_predictions if has_ground_truth else None
                    ),
                    "responses": responses,
                    "response_tokens": [
                        response_token_count(tokenizer, response) for response in responses
                    ],
                }
                results.append(result)
                output.write(json.dumps(result, ensure_ascii=False) + "\n")
                output.flush()

            del generated, sequences, gen_tokens, all_avg_logprobs
            torch.cuda.empty_cache()

    elapsed = time.time() - started
    print(f"samples={len(results)} generations={len(results) * args.num_samples}")
    print(f"elapsed_seconds={elapsed:.1f}")
    if has_ground_truth:
        baseline_correct = sum(row["baseline_correct"] for row in results)
        sample_oracle = sum(row["any_sample_correct"] for row in results)
        print(f"baseline={baseline_correct}/{len(results)} sample_oracle={sample_oracle}/{len(results)}")


if __name__ == "__main__":
    main()
