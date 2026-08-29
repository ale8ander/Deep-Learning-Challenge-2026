import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = Path("/workspace/models/Qwen2.5-3B-Instruct")
SYSTEM_PROMPT = (
    "You are a rigorous math-solution judge. Compare Response A and Response B "
    "against the problem, checking the reasoning and final integer. Choose the "
    "response that is mathematically correct. Reply with exactly A or B."
)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl(path: Path) -> dict[str, dict]:
    rows = {}
    with path.open(encoding="utf-8") as file:
        for line in file:
            if not line.strip():
                continue
            row = json.loads(line)
            problem_id = str(row["id"]).strip()
            if problem_id in rows or not isinstance(row.get("response"), str):
                raise SystemExit(f"Invalid or duplicate row in {path}: {problem_id}")
            rows[problem_id] = row
    return rows


def normalize(value) -> str:
    return str(value).strip().replace(",", "")


def shorten(text: str, max_chars: int) -> str:
    if len(text) <= max_chars:
        return text
    head = max_chars * 3 // 4
    return text[:head] + "\n[...middle omitted...]\n" + text[-(max_chars - head) :]


def choose_vote(votes: list[str], fallback_index: int) -> tuple[str, str, int]:
    counts = Counter(votes)
    best = max(counts.values())
    winners = [answer for answer, count in counts.items() if count == best]
    if len(winners) == 1 and best >= 2:
        return winners[0], "majority", best
    fallback = votes[fallback_index]
    return fallback, "fallback", counts[fallback]


def pair_prompt(question: str, response_a: str, response_b: str) -> list[dict]:
    user = (
        f"Problem:\n{question}\n\nResponse A:\n{response_a}\n\n"
        f"Response B:\n{response_b}\n\n"
        "Which response is correct? Reply exactly A or B."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--predictions", type=Path, action="append", required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    parser.add_argument("--max-response-chars", type=int, default=5000)
    parser.add_argument("--fallback-voter", type=int, default=1)
    parser.add_argument("--max-support", type=int, default=2)
    parser.add_argument("--threshold", type=float, default=0.0)
    parser.add_argument(
        "--require-swap-consistency", action=argparse.BooleanOptionalAction, default=True
    )
    args = parser.parse_args()

    questions = read_csv(args.questions)
    mappings = [read_jsonl(path) for path in args.predictions]
    expected = {row["id"].strip() for row in questions}
    for path, mapping in zip(args.predictions, mappings, strict=True):
        if set(mapping) != expected:
            raise SystemExit(f"ID set mismatch in {path}")
    fallback_index = args.fallback_voter - 1
    if not 0 <= fallback_index < len(mappings):
        raise SystemExit("--fallback-voter is out of range")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    a_ids = tokenizer("A", add_special_tokens=False)["input_ids"]
    b_ids = tokenizer("B", add_special_tokens=False)["input_ids"]
    if len(a_ids) != 1 or len(b_ids) != 1:
        raise SystemExit(f"A/B must each be one token: A={a_ids} B={b_ids}")
    a_id, b_id = a_ids[0], b_ids[0]

    tasks = []
    states = {}
    for question_row in questions:
        problem_id = question_row["id"].strip()
        votes = [normalize(mapping[problem_id]["prediction"]) for mapping in mappings]
        current, decision, support = choose_vote(votes, fallback_index)
        representatives = {}
        for vote, mapping in zip(votes, mappings, strict=True):
            representatives.setdefault(
                vote,
                shorten(mapping[problem_id]["response"], args.max_response_chars),
            )
        states[problem_id] = {
            "answer": normalize(question_row.get("answer", "")),
            "votes": votes,
            "fallback_prediction": current,
            "decision": decision,
            "support": support,
            "comparisons": [],
        }
        if support > args.max_support:
            continue
        for alternative, alternative_response in representatives.items():
            if alternative == current:
                continue
            current_response = representatives[current]
            for swapped, messages in (
                (False, pair_prompt(question_row["question"], current_response, alternative_response)),
                (True, pair_prompt(question_row["question"], alternative_response, current_response)),
            ):
                tasks.append(
                    {
                        "problem_id": problem_id,
                        "alternative": alternative,
                        "swapped": swapped,
                        "messages": messages,
                    }
                )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, local_files_only=True
    )
    model = PeftModel.from_pretrained(model, args.adapter_path, local_files_only=True)
    model = model.cuda().eval()
    raw_scores = {}
    with torch.inference_mode():
        for start in tqdm(range(0, len(tasks), args.batch_size), desc="selector-apply"):
            batch = tasks[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    task["messages"], tokenize=False, add_generation_prompt=True
                )
                for task in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            lengths = encoded["attention_mask"].sum(1).tolist()
            if max(lengths) > args.max_length:
                raise SystemExit(
                    f"Prompt exceeds --max-length in batch {start}: {max(lengths)}"
                )
            logits = model(**encoded.to(model.device), logits_to_keep=1).logits[:, -1, :].float()
            margins = (logits[:, a_id] - logits[:, b_id]).tolist()
            for task, margin in zip(batch, margins, strict=True):
                key = (task["problem_id"], task["alternative"])
                raw_scores.setdefault(key, {})["swapped" if task["swapped"] else "forward"] = margin

    fallback_correct = 0
    selected_correct = 0
    overrides = 0
    output_rows = []
    for question_row in questions:
        problem_id = question_row["id"].strip()
        state = states[problem_id]
        current = state["fallback_prediction"]
        best_alternative = None
        best_margin = float("inf")
        comparisons = []
        for (task_id, alternative), scores in raw_scores.items():
            if task_id != problem_id:
                continue
            forward = scores["forward"]
            swapped = scores["swapped"]
            current_margin = (forward - swapped) / 2
            consistent = forward < 0 and swapped > 0
            comparisons.append(
                {
                    "alternative": alternative,
                    "current_minus_alternative_margin": current_margin,
                    "forward_a_minus_b": forward,
                    "swapped_a_minus_b": swapped,
                    "alternative_preferred_consistently": consistent,
                }
            )
            eligible = current_margin < -args.threshold
            if args.require_swap_consistency:
                eligible = eligible and consistent
            if eligible and current_margin < best_margin:
                best_margin = current_margin
                best_alternative = alternative
        prediction = best_alternative if best_alternative is not None else current
        answer = state["answer"]
        fallback_correct += int(bool(answer) and current == answer)
        selected_correct += int(bool(answer) and prediction == answer)
        overrides += int(prediction != current)
        output_rows.append(
            {
                "id": problem_id,
                "answer": answer or None,
                "prediction": prediction,
                "correct": prediction == answer if answer else None,
                "fallback_prediction": current,
                "fallback_correct": current == answer if answer else None,
                "votes": state["votes"],
                "vote_decision": state["decision"],
                "winning_support": state["support"],
                "overridden": prediction != current,
                "comparisons": comparisons,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in output_rows:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"samples={len(output_rows)} pair_tasks={len(tasks)} overrides={overrides}")
    if all(state["answer"] for state in states.values()):
        print(
            f"fallback={fallback_correct}/{len(output_rows)} "
            f"selected={selected_correct}/{len(output_rows)} delta={selected_correct - fallback_correct:+d}"
        )
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
