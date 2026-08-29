import argparse
import csv
import json
import re
from pathlib import Path

from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "Solve the math problem independently. Give a concise, logically complete derivation. "
    "Do not restate the problem or explore multiple approaches. The answer is always an "
    "integer. End with exactly: Final answer: <integer>"
)


def read_jsonl(path: Path) -> list[dict]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def response_text(body: dict) -> str:
    chunks = []
    for output in body.get("output", []):
        if output.get("type") != "message":
            continue
        for content in output.get("content", []):
            if content.get("type") == "output_text" and content.get("text"):
                chunks.append(content["text"])
    return "\n".join(chunks).strip()


def extract_answer(text: str) -> str | None:
    matches = re.findall(
        r"final\s+answer\s*:\s*\$?\s*(-?\d[\d,]*)",
        text,
        flags=re.IGNORECASE,
    )
    if matches:
        return matches[-1].replace(",", "")
    boxed = re.findall(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}", text)
    return boxed[-1].replace(",", "") if boxed else None


def read_official_questions(data_dir: Path) -> dict[str, str]:
    path = data_dir / "deep_chal_math_train.csv"
    with path.open(encoding="utf-8-sig", newline="") as file:
        return {row["id"]: row["question"] for row in csv.DictReader(file)}


def normalize_api_results(
    rows: list[dict],
    baseline_by_id: dict[str, dict],
    official_questions: dict[str, str],
) -> list[dict]:
    if not rows or "custom_id" not in rows[0]:
        return rows
    normalized = []
    api_errors = 0
    for result in rows:
        problem_id = result.get("custom_id")
        if problem_id not in baseline_by_id:
            raise SystemExit(f"API result ID absent from baseline: {problem_id}")
        response = result.get("response") or {}
        if result.get("error") or response.get("status_code") != 200:
            api_errors += 1
            continue
        body = response.get("body") or {}
        text = response_text(body)
        prediction = extract_answer(text)
        old = baseline_by_id[problem_id]
        official_answer = str(old["answer"]).strip()
        normalized.append({
            "id": problem_id,
            "question": old.get("question") or official_questions.get(problem_id),
            "official_answer": official_answer,
            "prediction": prediction,
            "correct": prediction == official_answer,
            "response": text,
        })
    print(f"api_results={len(rows)} api_success={len(normalized)} api_errors={api_errors}")
    return normalized


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("baseline", type=Path)
    parser.add_argument("candidate", type=Path)
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/workspace/models/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--token-limit", type=int, default=512)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--intersection",
        action="store_true",
        help="Compare candidate IDs that also exist in the baseline.",
    )
    parser.add_argument(
        "--failed-output",
        type=Path,
        help="Write candidate failures as generator-compatible JSONL.",
    )
    parser.add_argument(
        "--accepted-sft-output",
        type=Path,
        help="Write correct candidate responses as three-message SFT JSONL.",
    )
    parser.add_argument(
        "--teacher",
        help="Teacher name required with --accepted-sft-output.",
    )
    parser.add_argument("--show-ids-limit", type=int, default=20)
    args = parser.parse_args()

    if args.accepted_sft_output is not None and not args.teacher:
        raise SystemExit("--teacher is required with --accepted-sft-output")

    baseline = read_jsonl(args.baseline)
    candidate = read_jsonl(args.candidate)
    baseline_by_id = {row["id"]: row for row in baseline}
    candidate = normalize_api_results(
        candidate,
        baseline_by_id,
        read_official_questions(args.data_dir),
    )
    candidate_by_id = {row["id"]: row for row in candidate}

    if len(baseline_by_id) != len(baseline) or len(candidate_by_id) != len(candidate):
        raise SystemExit("Duplicate IDs found")
    if args.intersection:
        unexpected = candidate_by_id.keys() - baseline_by_id.keys()
        if unexpected:
            raise SystemExit(
                f"Candidate contains {len(unexpected)} IDs absent from baseline"
            )
        if not candidate_by_id:
            raise SystemExit("Candidate is empty")
        baseline_by_id = {
            problem_id: baseline_by_id[problem_id]
            for problem_id in candidate_by_id
        }
        baseline = list(baseline_by_id.values())
    elif baseline_by_id.keys() != candidate_by_id.keys():
        raise SystemExit("The two files do not contain the same IDs")
    if not all("correct" in row for row in baseline + candidate):
        raise SystemExit("Both files must contain validation labels and correct fields")

    def answer(row: dict) -> str | None:
        value = row.get("answer", row.get("official_answer"))
        return str(value).strip() if value is not None else None

    for problem_id, old in baseline_by_id.items():
        new = candidate_by_id[problem_id]
        if answer(old) != answer(new):
            raise SystemExit(f"Official answer mismatch for {problem_id}")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    baseline_correct = sum(bool(row["correct"]) for row in baseline)
    candidate_correct = sum(bool(row["correct"]) for row in candidate)
    recovered = []
    regressed = []
    changed_wrong = []
    truncated = []

    for problem_id, old in baseline_by_id.items():
        new = candidate_by_id[problem_id]
        token_count = len(
            tokenizer(old["response"], add_special_tokens=False)["input_ids"]
        )
        if token_count >= args.token_limit - 2:
            truncated.append(problem_id)
        if not old["correct"] and new["correct"]:
            recovered.append(problem_id)
        elif old["correct"] and not new["correct"]:
            regressed.append(problem_id)
        elif not old["correct"] and not new["correct"] and old["prediction"] != new["prediction"]:
            changed_wrong.append(problem_id)

    total = len(baseline_by_id)
    print(f"samples={total}")
    print(f"baseline={baseline_correct}/{total} ({baseline_correct / total:.6f})")
    print(f"candidate={candidate_correct}/{total} ({candidate_correct / total:.6f})")
    print(f"delta={(candidate_correct - baseline_correct) / total:+.6f}")
    print(f"baseline_truncated={len(truncated)}")
    shown_recovered = recovered[: args.show_ids_limit]
    shown_regressed = regressed[: args.show_ids_limit]
    print(f"recovered={len(recovered)} sample_ids={','.join(shown_recovered)}")
    print(f"regressed={len(regressed)} sample_ids={','.join(shown_regressed)}")
    print(f"changed_but_still_wrong={len(changed_wrong)}")

    if args.failed_output is not None:
        failures = [
            candidate_by_id[problem_id]
            for problem_id in baseline_by_id
            if not candidate_by_id[problem_id]["correct"]
        ]
        args.failed_output.parent.mkdir(parents=True, exist_ok=True)
        with args.failed_output.open("w", encoding="utf-8") as file:
            for row in failures:
                item = {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": answer(row),
                    "previous_prediction": row.get("prediction"),
                    "previous_response": row.get("response"),
                }
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(f"failed_output={args.failed_output} failures={len(failures)}")

    if args.accepted_sft_output is not None:
        accepted = [
            candidate_by_id[problem_id]
            for problem_id in baseline_by_id
            if candidate_by_id[problem_id]["correct"]
        ]
        args.accepted_sft_output.parent.mkdir(parents=True, exist_ok=True)
        with args.accepted_sft_output.open("w", encoding="utf-8") as file:
            for row in accepted:
                item = {
                    "id": row["id"],
                    "question": row["question"],
                    "answer": answer(row),
                    "teacher": args.teacher,
                    "messages": [
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": row["question"]},
                        {"role": "assistant", "content": row["response"]},
                    ],
                }
                file.write(json.dumps(item, ensure_ascii=False) + "\n")
        print(
            f"accepted_sft_output={args.accepted_sft_output} "
            f"accepted={len(accepted)} teacher={args.teacher}"
        )


if __name__ == "__main__":
    main()
