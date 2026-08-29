import argparse
import json
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer


DEFAULT_MODEL_PATH = Path("/workspace/models/Qwen2.5-3B-Instruct")


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--adapter-path", type=Path, required=True)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-length", type=int, default=4096)
    args = parser.parse_args()

    rows = read_jsonl(args.data)
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
    model = PeftModel.from_pretrained(model, args.adapter_path, local_files_only=True)
    model = model.cuda().eval()

    correct = 0
    results = []
    with torch.inference_mode():
        for start in tqdm(range(0, len(rows), args.batch_size), desc="selector-dev"):
            batch = rows[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    row["messages"][:-1], tokenize=False, add_generation_prompt=True
                )
                for row in batch
            ]
            encoded = tokenizer(prompts, return_tensors="pt", padding=True)
            lengths = encoded["attention_mask"].sum(1).tolist()
            if max(lengths) > args.max_length:
                raise SystemExit(
                    f"Prompt exceeds --max-length in batch starting {start}: {max(lengths)}"
                )
            encoded = encoded.to(model.device)
            logits = model(**encoded, logits_to_keep=1).logits[:, -1, :].float()
            margins = logits[:, a_id] - logits[:, b_id]
            for row, margin in zip(batch, margins.tolist()):
                label = row["messages"][-1]["content"].strip()
                prediction = "A" if margin >= 0 else "B"
                is_correct = prediction == label
                correct += int(is_correct)
                results.append(
                    {
                        "id": row["id"],
                        "problem_id": row.get("problem_id"),
                        "label": label,
                        "prediction": prediction,
                        "a_minus_b_logit": margin,
                        "correct": is_correct,
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"samples={len(rows)} correct={correct} accuracy={correct / len(rows):.6f}")
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
