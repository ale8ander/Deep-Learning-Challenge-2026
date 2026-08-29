import argparse
import csv
import json
import re
import time
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from ensemble_predictions import normalize_integer
from submit_baseline import SYSTEM_PROMPTS, response_token_count


DEFAULT_MODEL_PATH = Path("/workspace/models/Qwen2.5-3B-Instruct")
FINAL_RE = re.compile(
    r"(?:final\s+answer|최종\s*정답)\s*(?:is|:|=)?\s*"
    r"(?:\\boxed\s*\{\s*)?(-?\d[\d,]*)",
    re.IGNORECASE,
)
RESCUE_SYSTEM_PROMPT = (
    "Solve the math problem carefully but concisely. Keep the entire solution under "
    "1000 tokens. Do not restate the problem and do not repeat a calculation or "
    "argument. If an approach begins to loop, abandon it immediately and use a "
    "shorter method. Check the result once. The answer is always an integer. "
    "End your response with exactly: Final answer: <integer>"
)
RESCUE_SYSTEM_PROMPT_2 = (
    "Solve this problem again from scratch using a structural or algebraic method. "
    "Avoid exhaustive listing, long case-by-case enumeration, and repeated equations. "
    "Keep the reasoning under 1000 tokens and focus on the exact quantity requested. "
    "Use a different route if the first idea becomes long. Verify the integer once. "
    "End your response with exactly: Final answer: <integer>"
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl(path: Path) -> dict[str, dict]:
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


def explicit_final(text: str) -> str | None:
    matches = FINAL_RE.findall(text)
    return normalize_integer(matches[-1].replace(",", "")) if matches else None


def max_repeated_ngram(text: str, size: int = 8) -> int:
    words = text.split()
    if len(words) < size:
        return 0
    counts = Counter(tuple(words[index : index + size]) for index in range(len(words) - size + 1))
    return max(counts.values(), default=0)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--stuck-threshold", type=int, default=2046)
    parser.add_argument("--max-new-tokens", type=int, default=4096)
    parser.add_argument("--batch-size", type=int, default=2)
    parser.add_argument("--repetition-penalty", type=float, default=1.05)
    parser.add_argument("--no-repeat-ngram-size", type=int, default=12)
    parser.add_argument(
        "--prompt-style",
        choices=("concise", "concise2", "default", "verify"),
        default="concise",
    )
    parser.add_argument("--gate-with", type=Path)
    parser.add_argument("--analyze-only", action="store_true")
    args = parser.parse_args()

    questions = read_csv(args.questions)
    question_by_id = {str(row["id"]).strip(): row for row in questions}
    predictions = read_jsonl(args.predictions)
    if set(question_by_id) != set(predictions):
        raise SystemExit("Question and prediction ID sets differ")

    stuck = []
    for problem_id, prediction_row in predictions.items():
        if int(prediction_row.get("response_tokens", 0)) < args.stuck_threshold:
            continue
        question_row = question_by_id[problem_id]
        answer = normalize_integer(question_row.get("answer"))
        old_prediction = normalize_integer(prediction_row.get("prediction"))
        response = str(prediction_row.get("response", ""))
        stuck.append(
            {
                "id": problem_id,
                "question": question_row["question"],
                "answer": answer,
                "old_prediction": old_prediction,
                "old_correct": old_prediction == answer if answer is not None else None,
                "old_response_tokens": int(prediction_row["response_tokens"]),
                "old_explicit_final": explicit_final(response),
                "old_max_repeat8": max_repeated_ngram(response),
            }
        )
    stuck.sort(key=lambda row: row["id"])

    print(f"rows={len(questions)} stuck={len(stuck)} threshold={args.stuck_threshold}")
    for row in stuck:
        print(
            f"id={row['id']} tokens={row['old_response_tokens']} "
            f"old={row['old_prediction']} answer={row['answer']} "
            f"correct={row['old_correct']} explicit={row['old_explicit_final']} "
            f"max_repeat8={row['old_max_repeat8']}"
        )
    if args.analyze_only:
        return
    if not args.adapter_path or not args.output:
        raise SystemExit("--adapter-path and --output are required unless --analyze-only")
    if not stuck:
        raise SystemExit("No stuck rows found")

    torch.manual_seed(20260831)
    torch.cuda.manual_seed_all(20260831)
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
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None
    if args.prompt_style == "concise":
        system_prompt = RESCUE_SYSTEM_PROMPT
    elif args.prompt_style == "concise2":
        system_prompt = RESCUE_SYSTEM_PROMPT_2
    else:
        system_prompt = SYSTEM_PROMPTS[args.prompt_style]

    results = []
    started = time.time()
    with torch.inference_mode():
        for start in tqdm(range(0, len(stuck), args.batch_size), desc="rescue-stuck"):
            batch = stuck[start : start + args.batch_size]
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
                do_sample=False,
                max_new_tokens=args.max_new_tokens,
                repetition_penalty=args.repetition_penalty,
                no_repeat_ngram_size=args.no_repeat_ngram_size,
                pad_token_id=tokenizer.eos_token_id,
            )
            responses = tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            for row, response in zip(batch, responses, strict=True):
                new_tokens = response_token_count(tokenizer, response)
                new_prediction = explicit_final(response)
                accepted = (
                    new_prediction is not None
                    and new_tokens < args.max_new_tokens - 2
                )
                selected_prediction = new_prediction if accepted else row["old_prediction"]
                result = {
                    **row,
                    "new_prediction": new_prediction,
                    "new_response_tokens": new_tokens,
                    "new_max_repeat8": max_repeated_ngram(response),
                    "accepted": accepted,
                    "selected_prediction": selected_prediction,
                    "new_correct": (
                        new_prediction == row["answer"]
                        if row["answer"] is not None and new_prediction is not None
                        else None
                    ),
                    "selected_correct": (
                        selected_prediction == row["answer"]
                        if row["answer"] is not None
                        else None
                    ),
                    "response": response,
                }
                results.append(result)

    if args.gate_with is not None:
        prior = read_jsonl(args.gate_with)
        if set(prior) != {row["id"] for row in results}:
            raise SystemExit("Gate result ID set differs")
        for row in results:
            prior_row = prior[row["id"]]
            agreement_accepted = (
                bool(prior_row.get("accepted"))
                and bool(row["accepted"])
                and normalize_integer(prior_row.get("new_prediction"))
                == row["new_prediction"]
            )
            row["single_selected_prediction"] = row["selected_prediction"]
            row["agreement_accepted"] = agreement_accepted
            row["selected_prediction"] = (
                row["new_prediction"] if agreement_accepted else row["old_prediction"]
            )
            row["selected_correct"] = (
                row["selected_prediction"] == row["answer"]
                if row["answer"] is not None
                else None
            )

    old_correct = sum(row["old_correct"] is True for row in results)
    selected_correct = sum(row["selected_correct"] is True for row in results)
    gains = sum(
        row["old_correct"] is False and row["selected_correct"] is True for row in results
    )
    regressions = sum(
        row["old_correct"] is True and row["selected_correct"] is False for row in results
    )
    accepted = sum(
        row.get("agreement_accepted", row["accepted"]) for row in results
    )
    elapsed = time.time() - started
    print(
        f"accepted={accepted}/{len(results)} old_correct={old_correct}/{len(results)} "
        f"selected_correct={selected_correct}/{len(results)} "
        f"delta={selected_correct - old_correct:+d} gains={gains} regressions={regressions}"
    )
    print(f"elapsed_seconds={elapsed:.1f}")
    for row in results:
        print(
            f"result id={row['id']} old={row['old_prediction']} "
            f"new={row['new_prediction']} selected={row['selected_prediction']} "
            f"answer={row['answer']} old_correct={row['old_correct']} "
            f"new_correct={row['new_correct']} tokens={row['new_response_tokens']} "
            f"repeat8={row['new_max_repeat8']} "
            f"accepted={row.get('agreement_accepted', row['accepted'])}"
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
