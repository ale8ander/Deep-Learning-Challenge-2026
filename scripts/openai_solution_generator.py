import argparse
import csv
import hashlib
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path


SYSTEM_PROMPT = (
    "Solve the math problem independently. Give a concise, logically complete derivation. "
    "Do not restate the problem or explore multiple approaches. The answer is always an "
    "integer. End with exactly: Final answer: <integer>"
)
ANSWER_CONDITIONED_SYSTEM_PROMPT = (
    "Create a concise, logically complete solution using the supplied official integer "
    "answer as a target to verify, not as permission to invent assumptions. Independently "
    "check every step. If the problem is ambiguous, corrupted, missing information, or the "
    "official answer cannot be validly derived, output exactly: REJECTED. Otherwise, do not "
    "restate the problem or explore multiple approaches. End with exactly: Final answer: "
    "<integer>"
)
# Verbose variant: the terse prompts above produced ~222-character median solutions that
# assert correct steps without showing how they were found. A 3B student needs the
# intermediate reasoning, the setup justification, and an explicit verification pass.
VERBOSE_SYSTEM_PROMPT = (
    "Solve the math problem independently, writing the solution as a teacher would for a "
    "student who must learn the reasoning process, not just the result.\n"
    "Requirements:\n"
    "1. State which approach you choose and why it fits this problem before applying it.\n"
    "2. Show every intermediate step explicitly. Never skip algebra or assert a value "
    "without deriving it. If a correspondence, case split, or substitution is used, explain "
    "how it was determined.\n"
    "3. When the problem needs case analysis, enumerate all cases and resolve each one.\n"
    "4. After reaching a candidate answer, verify it independently by substituting back "
    "into the original constraints or recomputing by a different route. Show this check.\n"
    "5. If the verification contradicts the candidate, correct the solution rather than "
    "keeping the original answer.\n"
    "The answer is always an integer. End with exactly: Final answer: <integer>"
)
VERBOSE_ANSWER_CONDITIONED_SYSTEM_PROMPT = (
    "Write a complete teaching solution for the problem, using the supplied official "
    "integer answer only as a target to verify, never as permission to invent assumptions "
    "or work backwards from the answer.\n"
    "Requirements:\n"
    "1. State which approach you choose and why it fits this problem before applying it.\n"
    "2. Show every intermediate step explicitly. Never skip algebra or assert a value "
    "without deriving it. If a correspondence, case split, or substitution is used, explain "
    "how it was determined.\n"
    "3. When the problem needs case analysis, enumerate all cases and resolve each one.\n"
    "4. After reaching your answer, verify it independently by substituting back into the "
    "original constraints or recomputing by a different route. Show this check.\n"
    "5. If the problem is ambiguous, corrupted, missing information, or the official answer "
    "cannot be validly derived, output exactly: REJECTED.\n"
    "The answer is always an integer. End with exactly: Final answer: <integer>"
)

STANDARD_PRICES_USD_PER_MILLION = {
    "gpt-5.6-luna": (0.20, 1.20),
    "gpt-5.4-mini": (0.75, 4.50),
    "gpt-5.4-mini-2026-03-17": (0.75, 4.50),
    "gpt-5.6-terra": (2.00, 12.00),
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def prices(model: str, pricing: str) -> tuple[float, float]:
    if model not in STANDARD_PRICES_USD_PER_MILLION:
        raise SystemExit(f"No recorded pricing for model: {model}")
    input_rate, output_rate = STANDARD_PRICES_USD_PER_MILLION[model]
    if pricing == "batch":
        return input_rate / 2, output_rate / 2
    return input_rate, output_rate


def is_validation_id(problem_id: str, ratio: float = 0.1) -> bool:
    value = int(hashlib.sha256(problem_id.encode()).hexdigest()[:8], 16) / 2**32
    return value < ratio


def official_training_rows(data_dir: Path) -> list[dict[str, str]]:
    rows = read_csv(data_dir / "deep_chal_math_train.csv")
    excluded = {row["id"] for row in read_csv(data_dir / "train_filtered_ids.csv")}
    clean = [row for row in rows if row["id"] not in excluded]
    training = [row for row in clean if not is_validation_id(row["id"])]
    # 663 excluded since the 36 label-error/malformed holdout problems were added on 2026-08-26.
    if (len(rows), len(excluded), len(clean), len(training)) != (17000, 663, 16337, 14703):
        raise SystemExit(
            f"Unexpected counts: total={len(rows)} excluded={len(excluded)} "
            f"clean={len(clean)} training={len(training)}"
        )
    return training


def selected_rows(data_dir: Path, limit: int, seed: int) -> list[dict[str, str]]:
    rows = official_training_rows(data_dir)
    if not 1 <= limit <= len(rows):
        raise SystemExit(f"--limit must be between 1 and {len(rows)}")
    return sorted(
        rows,
        key=lambda row: hashlib.sha256(f"{seed}:{row['id']}".encode()).hexdigest(),
    )[:limit]


def input_rows(data_dir: Path, input_jsonl: Path) -> list[dict[str, str]]:
    official = {row["id"]: row for row in official_training_rows(data_dir)}
    supplied = read_jsonl(input_jsonl)
    if not supplied:
        raise SystemExit(f"Empty --input-jsonl: {input_jsonl}")
    rows = []
    seen = set()
    for line_number, item in enumerate(supplied, start=1):
        problem_id = item.get("id")
        if problem_id not in official:
            raise SystemExit(
                f"Unexpected, excluded, or validation ID at "
                f"{input_jsonl}:{line_number}: {problem_id}"
            )
        if problem_id in seen:
            raise SystemExit(f"Duplicate ID in --input-jsonl: {problem_id}")
        expected = official[problem_id]
        if item.get("question") != expected["question"]:
            raise SystemExit(f"Question mismatch in --input-jsonl: {problem_id}")
        supplied_answer = str(item.get("answer", "")).strip()
        if supplied_answer != expected["answer"].strip():
            raise SystemExit(f"Answer mismatch in --input-jsonl: {problem_id}")
        seen.add(problem_id)
        rows.append(expected)
    return rows


def client():
    if not os.environ.get("OPENAI_API_KEY"):
        raise SystemExit("Set OPENAI_API_KEY in the shell; never store it in source files.")
    try:
        from openai import OpenAI
    except ImportError as error:
        raise SystemExit("Install first: pip install 'openai>=2,<3'") from error
    return OpenAI()


def prepare(args) -> None:
    rows = (
        input_rows(args.data_dir, args.input_jsonl)
        if args.input_jsonl is not None
        else selected_rows(args.data_dir, args.limit, args.seed)
    )
    failed_ids = set()
    for path in args.include_failed_from:
        for result in read_jsonl(path):
            if not result.get("correct", False):
                failed_ids.add(result["id"])
    if failed_ids:
        selected_ids = {row["id"] for row in rows}
        unexpected = failed_ids - selected_ids
        if unexpected:
            raise SystemExit(
                f"{len(unexpected)} failed IDs are outside the selected --limit range"
            )
        rows = [row for row in rows if row["id"] in failed_ids]
    excluded_ids = set()
    for path in args.exclude_requests:
        for request in read_jsonl(path):
            excluded_ids.add(request["custom_id"])
    rows = [row for row in rows if row["id"] not in excluded_ids]
    if args.request_limit is not None:
        if args.request_limit <= 0:
            raise SystemExit("--request-limit must be positive")
        rows = rows[: args.request_limit]
    if not rows:
        raise SystemExit("No requests remain after applying --exclude-requests")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    if getattr(args, "verbose_prompt", False):
        instructions = (
            VERBOSE_ANSWER_CONDITIONED_SYSTEM_PROMPT
            if args.answer_conditioned
            else VERBOSE_SYSTEM_PROMPT
        )
    else:
        instructions = (
            ANSWER_CONDITIONED_SYSTEM_PROMPT if args.answer_conditioned else SYSTEM_PROMPT
        )
    with args.output.open("w", encoding="utf-8") as file:
        for row in rows:
            request_input = (
                f"Problem:\n{row['question']}\n\n"
                f"Official integer answer: {row['answer'].strip()}"
                if args.answer_conditioned else row["question"]
            )
            request = {
                "custom_id": row["id"], "method": "POST", "url": "/v1/responses",
                "body": {
                    "model": args.model, "instructions": instructions,
                    "input": request_input,
                    "reasoning": {"effort": args.reasoning_effort},
                    "max_output_tokens": args.max_output_tokens,
                },
            }
            file.write(json.dumps(request, ensure_ascii=False) + "\n")
    _, batch_output_rate = prices(args.model, "batch")
    _, standard_output_rate = prices(args.model, "standard")
    batch_ceiling = len(rows) * args.max_output_tokens / 1_000_000 * batch_output_rate
    standard_ceiling = len(rows) * args.max_output_tokens / 1_000_000 * standard_output_rate
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(), "model": args.model,
        "reasoning_effort": args.reasoning_effort, "max_output_tokens": args.max_output_tokens,
        "answer_conditioned": args.answer_conditioned,
        "seed": args.seed if args.input_jsonl is None else None,
        "selection_limit": args.limit if args.input_jsonl is None else None,
        "input_jsonl": str(args.input_jsonl) if args.input_jsonl else None,
        "input_jsonl_sha256": sha256(args.input_jsonl) if args.input_jsonl else None,
        "failed_ids_selected": len(failed_ids),
        "request_limit": args.request_limit,
        "excluded_existing": len(excluded_ids), "requests_created": len(rows),
        "selection": (
            "explicit validated IDs from --input-jsonl"
            if args.input_jsonl is not None else
            "failed IDs from checked files within deterministic clean non-validation selection"
            if failed_ids else
            "lowest sha256(f'{seed}:{id}') from clean non-validation train split"
        ),
        "requests": str(args.output), "requests_sha256": sha256(args.output),
        "batch_output_cost_ceiling_usd": round(batch_ceiling, 6),
        "standard_output_cost_ceiling_usd": round(standard_ceiling, 6),
        "note": "Input cost is additional and small; no validation/leaderboard/test rows are used.",
    }
    args.output.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))
    print("Prepared only; no API request was submitted.")


def submit(args) -> None:
    requests = read_jsonl(args.requests)
    if not requests:
        raise SystemExit("Empty requests file")
    models = {row["body"]["model"] for row in requests}
    limits = {row["body"]["max_output_tokens"] for row in requests}
    if len(models) != 1 or len(limits) != 1:
        raise SystemExit("Requests must use one model and one token limit")
    model = next(iter(models))
    _, output_rate = prices(model, "batch")
    ceiling = len(requests) * next(iter(limits)) / 1_000_000 * output_rate
    if ceiling > args.max_output_budget_usd:
        raise SystemExit(
            f"Refusing: output ceiling ${ceiling:.4f} exceeds budget "
            f"${args.max_output_budget_usd:.4f}"
        )
    api = client()
    with args.requests.open("rb") as file:
        uploaded = api.files.create(file=file, purpose="batch")
    batch = api.batches.create(
        input_file_id=uploaded.id, endpoint="/v1/responses", completion_window="24h",
        metadata={"project": "deep-chal-math-cot"},
    )
    state = {
        "submitted_at": datetime.now(timezone.utc).isoformat(),
        "requests": str(args.requests), "requests_sha256": sha256(args.requests),
        "requests_count": len(requests), "model": model,
        "max_output_tokens": next(iter(limits)), "strict_output_cost_ceiling_usd": ceiling,
        "input_file_id": uploaded.id, "batch_id": batch.id, "status": batch.status,
    }
    args.state.parent.mkdir(parents=True, exist_ok=True)
    args.state.write_text(json.dumps(state, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(state, indent=2))


def load_state(path: Path) -> dict:
    if not path.exists():
        raise SystemExit(f"State file not found: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def status(args) -> None:
    state = load_state(args.state)
    print(client().batches.retrieve(state["batch_id"]).model_dump_json(indent=2))


def cancel(args) -> None:
    state = load_state(args.state)
    batch = client().batches.cancel(state["batch_id"])
    print(batch.model_dump_json(indent=2))


def run_sync(args) -> None:
    from concurrent.futures import ThreadPoolExecutor, as_completed

    requests = read_jsonl(args.requests)
    if not requests:
        raise SystemExit("Empty requests file")
    limits = {row["body"]["max_output_tokens"] for row in requests}
    models = {row["body"]["model"] for row in requests}
    if len(limits) != 1 or len(models) != 1:
        raise SystemExit("Requests must use one model and one token limit")
    model = next(iter(models))
    _, output_rate = prices(model, "standard")
    ceiling = (
        len(requests) * next(iter(limits)) / 1_000_000
        * output_rate
    )
    if ceiling > args.max_output_budget_usd:
        raise SystemExit(
            f"Refusing: output ceiling ${ceiling:.4f} exceeds budget "
            f"${args.max_output_budget_usd:.4f}"
        )
    api = client()

    def invoke(request: dict) -> dict:
        try:
            response = api.responses.create(**request["body"])
            return {
                "custom_id": request["custom_id"],
                "response": {"status_code": 200, "body": response.model_dump(mode="json")},
                "error": None,
            }
        except Exception as error:
            return {
                "custom_id": request["custom_id"],
                "response": None,
                "error": {"type": type(error).__name__, "message": str(error)},
            }

    args.output.parent.mkdir(parents=True, exist_ok=True)
    completed = 0
    with args.output.open("w", encoding="utf-8") as file:
        with ThreadPoolExecutor(max_workers=args.workers) as pool:
            futures = [pool.submit(invoke, request) for request in requests]
            for future in as_completed(futures):
                result = future.result()
                file.write(json.dumps(result, ensure_ascii=False) + "\n")
                file.flush()
                completed += 1
                print(f"completed={completed}/{len(requests)} id={result['custom_id']}")
    print(f"results={args.output} output_cost_ceiling_usd={ceiling:.6f}")


def download(args) -> None:
    state = load_state(args.state)
    api = client()
    batch = api.batches.retrieve(state["batch_id"])
    if batch.status != "completed" or not batch.output_file_id:
        raise SystemExit(f"Batch is not ready: status={batch.status}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    api.files.content(batch.output_file_id).write_to_file(args.output)
    print(f"downloaded={args.output} sha256={sha256(args.output)}")
    if batch.error_file_id:
        error_path = args.output.with_name(args.output.stem + "_errors.jsonl")
        api.files.content(batch.error_file_id).write_to_file(error_path)
        print(f"errors={error_path}")


def response_text(body: dict) -> str:
    chunks = []
    for output in body.get("output", []):
        if output.get("type") == "message":
            for content in output.get("content", []):
                if content.get("type") == "output_text" and content.get("text"):
                    chunks.append(content["text"])
    return "\n".join(chunks).strip()


def extract_answer(text: str) -> str | None:
    matches = re.findall(
        r"final\s+answer\s*:\s*\$?\s*(-?\d[\d,]*)", text, flags=re.IGNORECASE
    )
    if matches:
        return matches[-1].replace(",", "")
    boxed = re.findall(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}", text)
    return boxed[-1].replace(",", "") if boxed else None


def process(args) -> None:
    official = {row["id"]: row for row in official_training_rows(args.data_dir)}
    results = read_jsonl(args.results)
    processed = []
    input_tokens = output_tokens = reasoning_tokens = api_errors = 0
    for result in results:
        problem_id = result.get("custom_id")
        if problem_id not in official:
            raise SystemExit(f"Unexpected or validation ID: {problem_id}")
        response = result.get("response") or {}
        if result.get("error") or response.get("status_code") != 200:
            api_errors += 1
            continue
        body = response.get("body") or {}
        text = response_text(body)
        prediction = extract_answer(text)
        answer = official[problem_id]["answer"].strip()
        usage = body.get("usage") or {}
        details = usage.get("output_tokens_details") or {}
        input_tokens += int(usage.get("input_tokens") or 0)
        output_tokens += int(usage.get("output_tokens") or 0)
        reasoning_tokens += int(details.get("reasoning_tokens") or 0)
        processed.append({
            "id": problem_id, "question": official[problem_id]["question"],
            "official_answer": answer, "prediction": prediction,
            "correct": prediction == answer, "response": text,
        })
    accepted = [row for row in processed if row["correct"]]
    args.raw_output.parent.mkdir(parents=True, exist_ok=True)
    with args.raw_output.open("w", encoding="utf-8") as file:
        for row in processed:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    args.sft_output.parent.mkdir(parents=True, exist_ok=True)
    with args.sft_output.open("w", encoding="utf-8") as file:
        for row in accepted:
            item = {
                "id": row["id"], "question": row["question"],
                "answer": row["official_answer"], "teacher": args.model,
                "messages": [
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": row["question"]},
                    {"role": "assistant", "content": row["response"]},
                ],
            }
            file.write(json.dumps(item, ensure_ascii=False) + "\n")
    input_rate, output_rate = prices(args.model, args.pricing)
    cost = input_tokens / 1_000_000 * input_rate + output_tokens / 1_000_000 * output_rate
    report = {
        "model": args.model, "results": len(results), "processed": len(processed),
        "api_errors": api_errors, "accepted": len(accepted),
        "acceptance_rate": len(accepted) / len(processed) if processed else 0,
        "input_tokens": input_tokens, "output_tokens": output_tokens,
        "reasoning_tokens_in_output": reasoning_tokens,
        "pricing": args.pricing,
        "estimated_api_cost_usd": round(cost, 6),
        "raw_output": str(args.raw_output), "raw_output_sha256": sha256(args.raw_output),
        "sft_output": str(args.sft_output), "sft_output_sha256": sha256(args.sft_output),
    }
    args.sft_output.with_suffix(".manifest.json").write_text(
        json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(json.dumps(report, ensure_ascii=False, indent=2))


def parser() -> argparse.ArgumentParser:
    root = argparse.ArgumentParser()
    commands = root.add_subparsers(dest="command", required=True)
    p = commands.add_parser("prepare")
    p.add_argument("--data-dir", type=Path, default=Path("data")); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--input-jsonl", type=Path)
    p.add_argument("--limit", type=int, default=10); p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--exclude-requests", type=Path, action="append", default=[])
    p.add_argument("--include-failed-from", type=Path, action="append", default=[])
    p.add_argument("--request-limit", type=int)
    p.add_argument("--model", default="gpt-5.6-luna")
    p.add_argument(
        "--reasoning-effort",
        choices=("none", "low", "medium", "high", "xhigh", "max"),
        default="low",
    )
    p.add_argument("--answer-conditioned", action="store_true")
    p.add_argument("--verbose-prompt", action="store_true")
    p.add_argument("--max-output-tokens", type=int, default=1024); p.set_defaults(function=prepare)
    p = commands.add_parser("submit")
    p.add_argument("--requests", type=Path, required=True); p.add_argument("--state", type=Path, required=True)
    p.add_argument("--max-output-budget-usd", type=float, default=0.01); p.set_defaults(function=submit)
    p = commands.add_parser("status"); p.add_argument("--state", type=Path, required=True); p.set_defaults(function=status)
    p = commands.add_parser("cancel"); p.add_argument("--state", type=Path, required=True); p.set_defaults(function=cancel)
    p = commands.add_parser("run-sync")
    p.add_argument("--requests", type=Path, required=True); p.add_argument("--output", type=Path, required=True)
    p.add_argument("--workers", type=int, default=5)
    p.add_argument("--max-output-budget-usd", type=float, default=0.02); p.set_defaults(function=run_sync)
    p = commands.add_parser("download")
    p.add_argument("--state", type=Path, required=True); p.add_argument("--output", type=Path, required=True); p.set_defaults(function=download)
    p = commands.add_parser("process")
    p.add_argument("--data-dir", type=Path, default=Path("data")); p.add_argument("--results", type=Path, required=True)
    p.add_argument("--raw-output", type=Path, required=True); p.add_argument("--sft-output", type=Path, required=True)
    p.add_argument("--model", default="gpt-5.6-luna")
    p.add_argument("--pricing", choices=("standard", "batch"), default="standard")
    p.set_defaults(function=process)
    return root


if __name__ == "__main__":
    args = parser().parse_args()
    args.function(args)
