import argparse
import json
import re
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer


DIGIT_TO_CATEGORY = {
    "1": "number_theory",
    "2": "combinatorics",
    "3": "algebra",
    "4": "geometry",
    "5": "other",
}
SYSTEM_PROMPT = (
    "Classify the primary mathematical domain. Reply with exactly one digit. "
    "1 = number theory: divisibility, congruences, primes, integer equations, "
    "digits or bases, arithmetic functions. "
    "2 = combinatorics: counting, permutations, subsets, probability, graphs, "
    "extremal set systems, or recurrences used primarily to count. "
    "3 = algebra: equations, inequalities, functions, polynomials, or sequences "
    "when integer arithmetic structure is not primary. "
    "4 = geometry: Euclidean, coordinate, or solid geometry. "
    "5 = other. "
    "Examples: 'remainder of 7^100 modulo 13' -> 1; "
    "'smallest integer with exactly 36 divisors' -> 1; "
    "'how many permutations avoid a pattern' -> 2; "
    "'largest subset with no forbidden pair' -> 2; "
    "'in how many orders can objects be removed from a grid' -> 2; "
    "'solve a polynomial equation over the reals' -> 3; "
    "'maximize an expression under a real constraint' -> 3; "
    "'find an angle in a triangle' -> 4; "
    "'a shopping or speed word problem with no counting structure' -> 5. "
    "Questions asking how many integers satisfy a divisibility, digit, base, "
    "remainder, or prime condition are number theory (1), while questions "
    "counting arrangements, selections, distributions, paths, games, or "
    "configurations are combinatorics (2). Output only 1, 2, 3, 4, or 5."
)


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/workspace/models/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--adapter-path", type=Path)
    parser.add_argument("--batch-size", type=int, default=16)
    args = parser.parse_args()

    rows = read_jsonl(args.input)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True
    )
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        dtype=torch.float16,
        local_files_only=True,
    )
    if args.adapter_path:
        model = PeftModel.from_pretrained(
            model, args.adapter_path, local_files_only=True
        )
    model = model.cuda().eval()
    model.generation_config.do_sample = False
    model.generation_config.temperature = None
    model.generation_config.top_p = None
    model.generation_config.top_k = None

    results = []
    with torch.inference_mode():
        for start in range(0, len(rows), args.batch_size):
            batch = rows[start : start + args.batch_size]
            prompts = [
                tokenizer.apply_chat_template(
                    [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": (
                                "Do not solve the problem. Classify only its primary domain.\n\n"
                                f"Problem:\n{row['question']}\n\n"
                                "Classification digit (1, 2, 3, 4, or 5):"
                            ),
                        },
                    ],
                    tokenize=False,
                    add_generation_prompt=True,
                )
                for row in batch
            ]
            encoded = tokenizer(
                prompts, return_tensors="pt", padding=True
            ).to(model.device)
            generated = model.generate(
                **encoded,
                max_new_tokens=3,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
            texts = tokenizer.batch_decode(
                generated[:, encoded["input_ids"].shape[1] :],
                skip_special_tokens=True,
            )
            for row, text in zip(batch, texts):
                match = re.search(r"[1-5]", text)
                digit = match.group(0) if match else None
                results.append(
                    {
                        "id": row["id"],
                        "question": row["question"],
                        "answer": str(row["answer"]),
                        "category": (
                            DIGIT_TO_CATEGORY[digit] if digit else "unparsed"
                        ),
                        "raw_classification": text.strip(),
                    }
                )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in results:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(dict(sorted(Counter(row["category"] for row in results).items())))
    print(f"output={args.output}")


if __name__ == "__main__":
    main()
