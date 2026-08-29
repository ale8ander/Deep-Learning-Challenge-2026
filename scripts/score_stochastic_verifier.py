import argparse
import csv
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

from build_stochastic_verifier_data import SYSTEM_PROMPT, shorten
from ensemble_predictions import normalize_integer


DEFAULT_MODEL_PATH = Path("/workspace/models/Qwen2.5-3B-Instruct")


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


def prompt_messages(question: str, response: str, prediction: str) -> list[dict[str, str]]:
    user = (
        f"Problem:\n{question}\n\n"
        f"Candidate solution:\n{response}\n\n"
        f"Candidate final answer: {prediction}\n\n"
        "Is the submitted final integer correct? Reply exactly A or B."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
    ]


def auc(labels: list[int], scores: list[float]) -> float | None:
    positives = sum(labels)
    negatives = len(labels) - positives
    if positives == 0 or negatives == 0:
        return None
    ordered = sorted(zip(scores, labels), key=lambda item: item[0])
    positive_rank_sum = 0.0
    index = 0
    while index < len(ordered):
        end = index + 1
        while end < len(ordered) and ordered[end][0] == ordered[index][0]:
            end += 1
        average_rank = ((index + 1) + end) / 2
        positive_rank_sum += average_rank * sum(label for _, label in ordered[index:end])
        index = end
    return (positive_rank_sum - positives * (positives + 1) / 2) / (positives * negatives)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--questions", type=Path, required=True)
    parser.add_argument("--samples", type=Path, required=True)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=2048)
    parser.add_argument("--max-response-chars", type=int, default=4000)
    args = parser.parse_args()

    questions = read_csv(args.questions)
    if not questions or not {"id", "question"}.issubset(questions[0]):
        raise SystemExit("Questions CSV must contain id and question")
    has_answers = "answer" in questions[0]
    question_ids = [row["id"].strip() for row in questions]
    expected = set(question_ids)
    samples = read_jsonl(args.samples)
    baseline = read_jsonl(args.baseline)
    for path, mapping in ((args.samples, samples), (args.baseline, baseline)):
        if set(mapping) != expected:
            raise SystemExit(f"ID set mismatch in {path}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    a_ids = tokenizer("A", add_special_tokens=False)["input_ids"]
    b_ids = tokenizer("B", add_special_tokens=False)["input_ids"]
    if len(a_ids) != 1 or len(b_ids) != 1:
        raise SystemExit(f"A/B must each be one token: A={a_ids} B={b_ids}")
    a_id, b_id = a_ids[0], b_ids[0]
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, local_files_only=True
    )
    model = PeftModel.from_pretrained(
        model, args.adapter_path, local_files_only=True
    ).cuda().eval()

    prepared = []
    for question_row in questions:
        problem_id = question_row["id"].strip()
        answer = normalize_integer(question_row["answer"]) if has_answers else None
        base = baseline[problem_id]
        base_prediction = normalize_integer(base.get("prediction"))
        base_response = base.get("response")
        if base_prediction is None or not isinstance(base_response, str):
            raise SystemExit(f"Invalid baseline response for {problem_id}")
        items = [
            {
                "source": "baseline",
                "index": 0,
                "prediction": base_prediction,
                "response": shorten(base_response, args.max_response_chars),
            }
        ]
        sample_row = samples[problem_id]
        responses = sample_row.get("responses")
        predictions = sample_row.get("sample_predictions")
        if not isinstance(responses, list) or not isinstance(predictions, list):
            raise SystemExit(f"Missing sample candidates for {problem_id}")
        if len(responses) != len(predictions):
            raise SystemExit(f"Sample length mismatch for {problem_id}")
        for index, (response, prediction) in enumerate(
            zip(responses, predictions, strict=True)
        ):
            normalized = normalize_integer(prediction)
            if normalized is None or not isinstance(response, str):
                continue
            items.append(
                {
                    "source": "sample",
                    "index": index,
                    "prediction": normalized,
                    "response": shorten(response, args.max_response_chars),
                }
            )
        prepared.append(
            {
                "id": problem_id,
                "question": question_row["question"],
                "answer": answer,
                "baseline_prediction": base_prediction,
                "items": items,
            }
        )

    flat = [(row, item) for row in prepared for item in row["items"]]
    labels: list[int] = []
    scores: list[float] = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(flat), args.batch_size), desc="score-verifier"):
            batch = flat[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    prompt_messages(row["question"], item["response"], item["prediction"]),
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row, item in batch
            ]
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True, truncation=False
            )
            lengths = encoded["attention_mask"].sum(1).tolist()
            if max(lengths) > args.max_length:
                raise SystemExit(
                    f"Prompt exceeds --max-length in batch {start}: {max(lengths)}"
                )
            encoded = encoded.to(model.device)
            logits = model(**encoded, logits_to_keep=1).logits[:, -1, :].float()
            margins = (logits[:, a_id] - logits[:, b_id]).tolist()
            for (row, item), margin in zip(batch, margins, strict=True):
                item["a_minus_b_logit"] = margin
                item.pop("response")
                if has_answers:
                    label = int(item["prediction"] == row["answer"])
                    item["correct"] = bool(label)
                    labels.append(label)
                    scores.append(margin)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in prepared:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    if has_answers:
        predicted = [int(score >= 0) for score in scores]
        tp = sum(prediction and label for prediction, label in zip(predicted, labels))
        tn = sum(not prediction and not label for prediction, label in zip(predicted, labels))
        positives = sum(labels)
        negatives = len(labels) - positives
        accuracy = (tp + tn) / len(labels)
        balanced = ((tp / positives) + (tn / negatives)) / 2
        print(
            f"candidates={len(labels)} accuracy={accuracy:.6f} "
            f"balanced_accuracy={balanced:.6f} auc={auc(labels, scores):.6f}"
        )
    print(f"problems={len(prepared)} output={args.output}")


if __name__ == "__main__":
    main()
