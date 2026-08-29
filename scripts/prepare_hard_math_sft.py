import argparse
from collections import Counter, defaultdict
from difflib import SequenceMatcher
import csv
import hashlib
import json
import re
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer


SYSTEM_PROMPT = (
    "Solve the math problem independently. Give a concise, logically complete derivation. "
    "Do not restate the problem or explore multiple approaches. The answer is always an "
    "integer. End with exactly: Final answer: <integer>"
)
GSM8K_DATASET = "openai/gsm8k"
GSM8K_REVISION = "740312a"
MATH_DATASET = "DigitalLearningGmbH/MATH-lighteval"
MATH_REVISION = "92ace7ed9c5f22d9148ea70c403948eae7bed2e8"
NUMINA_DATASET = "AI-MO/NuminaMath-1.5"
NUMINA_REVISION = "1b05109f9e5c1ad06c0663519502416c30b300f8"
NUMINA_SOURCE_QUOTAS = {
    "amc_aime": 150,
    "olympiads_ref": 75,
    "cn_contest": 50,
    "number_theory": 25,
}
NUMINA_BROAD_HARD_SOURCE_QUOTAS = {
    "olympiads": 600,
    "aops_forum": 450,
    "cn_contest": 300,
    "amc_aime": 100,
    "olympiads_ref": 50,
}
NUMINA_BROAD_10K_SOURCE_QUOTAS = {
    "olympiads": 3500,
    "aops_forum": 2200,
    "cn_contest": 1854,
    "amc_aime": 169,
    "olympiads_ref": 77,
}
MATH_LEVEL_QUOTAS = {"Level 3": 150, "Level 4": 275, "Level 5": 275}
VISUAL_PATTERN = re.compile(
    r"https?://|www\.|\\includegraphics|\.(?:png|jpe?g|gif)\b|"
    r"\b(?:diagram|figure)\s+(?:above|below)|\bshown\s+(?:above|below)\b",
    re.IGNORECASE,
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def stable_score(seed: int, namespace: str, text: str) -> str:
    return hashlib.sha256(f"{seed}:{namespace}:{text}".encode()).hexdigest()


def normalized_tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+|\d+", text.casefold())


def normalized_text(text: str) -> str:
    return " ".join(normalized_tokens(text))


def ngrams(tokens: list[str], width: int = 5) -> set[tuple[str, ...]]:
    if len(tokens) < width:
        return {tuple(tokens)} if tokens else set()
    return {tuple(tokens[index : index + width]) for index in range(len(tokens) - width + 1)}


class OfficialDecontaminator:
    def __init__(self, questions: list[str]) -> None:
        self.normalized = [normalized_text(question) for question in questions]
        self.exact = set(self.normalized)
        self.tokens = [value.split() for value in self.normalized]
        self.index: dict[tuple[str, ...], set[int]] = defaultdict(set)
        for index, tokens in enumerate(self.tokens):
            for gram in ngrams(tokens):
                self.index[gram].add(index)

    def match(self, question: str) -> str | None:
        normalized = normalized_text(question)
        if not normalized:
            return "empty"
        if normalized in self.exact:
            return "exact"
        tokens = normalized.split()
        candidates: set[int] = set()
        for gram in ngrams(tokens):
            candidates.update(self.index.get(gram, ()))
        token_set = set(tokens)
        for index in candidates:
            official_tokens = self.tokens[index]
            official_set = set(official_tokens)
            union = token_set | official_set
            jaccard = len(token_set & official_set) / len(union) if union else 1.0
            length_ratio = min(len(tokens), len(official_tokens)) / max(
                len(tokens), len(official_tokens)
            )
            if jaccard >= 0.85 and length_ratio >= 0.80:
                return "near_token"
            if length_ratio >= 0.85 and SequenceMatcher(
                None, normalized, self.normalized[index], autojunk=False
            ).ratio() >= 0.92:
                return "near_sequence"
        return None


def extract_braced(text: str, open_brace: int) -> str | None:
    depth = 0
    for index in range(open_brace, len(text)):
        if text[index] == "{":
            depth += 1
        elif text[index] == "}":
            depth -= 1
            if depth == 0:
                return text[open_brace + 1 : index]
    return None


def boxed_values(solution: str) -> list[str]:
    values = []
    for match in re.finditer(r"\\boxed\s*\{", solution):
        value = extract_braced(solution, match.end() - 1)
        if value is not None:
            values.append(value)
    return values


def parse_integer(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    text = text.replace(",", "").replace("−", "-").replace("–", "-")
    text = re.sub(r"\\(?:text|mathrm|operatorname)\s*\{[^{}]*\}", "", text)
    text = re.sub(r"(?:\^\s*\{?\\circ\}?|\\circ|°|\\%)$", "", text).strip()
    match = re.fullmatch(r"([+-]?\d+)(?:\.0+)?", text)
    if match:
        return str(int(match.group(1)))
    return None


def math_answer(solution: str) -> str | None:
    values = boxed_values(solution)
    return parse_integer(values[-1]) if values else None


def gsm8k_solution(answer_text: str) -> tuple[str, str] | None:
    match = re.search(r"####\s*([^\n]+)\s*$", answer_text)
    if match is None:
        return None
    answer = parse_integer(match.group(1))
    if answer is None:
        return None
    solution = re.sub(r"\s*####\s*[^\n]+\s*$", "", answer_text).strip()
    solution = re.sub(r"<<[^<>]*>>", "", solution)
    return solution, answer


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def valid_flag(value: object) -> bool:
    return value is True or str(value).strip().casefold() in {"yes", "true", "valid", "1"}


def build_sft_row(
    identifier: str,
    question: str,
    solution: str,
    answer: str,
    teacher: str,
    metadata: dict,
) -> dict:
    solution = solution.strip()
    solution = re.sub(r"\n*Final answer:\s*[+-]?\d+\s*$", "", solution).rstrip()
    assistant = f"{solution}\n\nFinal answer: {answer}"
    return {
        "id": identifier,
        "question": question.strip(),
        "answer": answer,
        "teacher": teacher,
        "metadata": metadata,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": question.strip()},
            {"role": "assistant", "content": assistant},
        ],
    }


def token_length(tokenizer, messages: list[dict]) -> int:
    return len(
        tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=False,
        )
    )


def filter_candidate(
    row: dict,
    tokenizer,
    decontaminator: OfficialDecontaminator,
    seen_questions: set[str],
    min_assistant_tokens: int,
    max_assistant_tokens: int,
    max_seq_length: int,
    rejections: Counter,
) -> bool:
    question = row["question"]
    assistant = row["messages"][-1]["content"]
    normalized = normalized_text(question)
    if normalized in seen_questions:
        rejections["external_duplicate"] += 1
        return False
    if VISUAL_PATTERN.search(question):
        rejections["visual_or_url"] += 1
        return False
    contamination = decontaminator.match(question)
    if contamination:
        rejections[f"official_{contamination}"] += 1
        return False
    assistant_tokens = len(tokenizer.encode(assistant, add_special_tokens=False))
    if assistant_tokens < min_assistant_tokens:
        rejections["solution_too_short"] += 1
        return False
    if assistant_tokens > max_assistant_tokens:
        rejections["solution_too_long"] += 1
        return False
    total_tokens = token_length(tokenizer, row["messages"])
    if total_tokens > max_seq_length:
        rejections["sequence_too_long"] += 1
        return False
    row["metadata"]["assistant_tokens"] = assistant_tokens
    row["metadata"]["sequence_tokens"] = total_tokens
    seen_questions.add(normalized)
    return True


def choose_by_quota(
    candidates: list[dict],
    quota_key: str,
    quotas: dict[str, int],
    seed: int,
    namespace: str,
) -> list[dict]:
    groups: dict[str, list[dict]] = defaultdict(list)
    for row in candidates:
        groups[row["metadata"][quota_key]].append(row)
    selected = []
    for key, count in quotas.items():
        ranked = sorted(
            groups.get(key, []),
            key=lambda row: stable_score(seed, f"{namespace}:{key}", row["question"]),
        )
        if len(ranked) < count:
            available = {group: len(rows) for group, rows in sorted(groups.items())}
            raise SystemExit(
                f"Not enough {namespace} rows for {key}: {len(ranked)} < {count}; "
                f"available={available}"
            )
        selected.extend(ranked[:count])
    return selected


def scale_quotas(quotas: dict[str, int], requested_count: int) -> dict[str, int]:
    if requested_count <= 0:
        raise SystemExit("Requested sample count must be positive")
    original_count = sum(quotas.values())
    exact = {
        key: requested_count * count / original_count for key, count in quotas.items()
    }
    scaled = {key: int(value) for key, value in exact.items()}
    remainder = requested_count - sum(scaled.values())
    priority = sorted(
        quotas,
        key=lambda key: (-(exact[key] - scaled[key]), key),
    )
    for key in priority[:remainder]:
        scaled[key] += 1
    return scaled


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/workspace/models/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/external/hard_math_1000.jsonl"),
    )
    parser.add_argument("--math-count", type=int, default=700)
    parser.add_argument("--numina-count", type=int, default=300)
    parser.add_argument(
        "--numina-profile",
        choices=("legacy", "broad_hard", "broad_10k"),
        default="legacy",
    )
    parser.add_argument("--gsm8k-count", type=int, default=0)
    parser.add_argument(
        "--exclude-data",
        type=Path,
        action="append",
        default=[],
        help="JSONL SFT data whose normalized questions must be excluded.",
    )
    parser.add_argument("--min-assistant-tokens", type=int, default=64)
    parser.add_argument("--max-assistant-tokens", type=int, default=900)
    parser.add_argument("--max-seq-length", type=int, default=1536)
    parser.add_argument("--seed", type=int, default=2026)
    parser.add_argument("--math-revision", default=MATH_REVISION)
    parser.add_argument("--numina-revision", default=NUMINA_REVISION)
    parser.add_argument("--gsm8k-revision", default=GSM8K_REVISION)
    args = parser.parse_args()

    math_level_quotas = scale_quotas(MATH_LEVEL_QUOTAS, args.math_count)
    numina_profile_quotas = {
        "legacy": NUMINA_SOURCE_QUOTAS,
        "broad_hard": NUMINA_BROAD_HARD_SOURCE_QUOTAS,
        "broad_10k": NUMINA_BROAD_10K_SOURCE_QUOTAS,
    }[args.numina_profile]
    numina_source_quotas = scale_quotas(numina_profile_quotas, args.numina_count)

    official_train = read_csv(args.data_dir / "deep_chal_math_train.csv")
    official_leaderboard = read_csv(
        args.data_dir / "deep_chal_math_leaderboard_filtered.csv"
    )
    official_questions = [row["question"] for row in official_train + official_leaderboard]
    decontaminator = OfficialDecontaminator(official_questions)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    seen_questions: set[str] = set()
    excluded_questions: set[str] = set()
    excluded_rows = 0
    rejections: Counter = Counter()
    for exclude_path in args.exclude_data:
        with exclude_path.open(encoding="utf-8") as file:
            for line_number, line in enumerate(file, start=1):
                if not line.strip():
                    continue
                row = json.loads(line)
                question = row.get("question")
                if not isinstance(question, str) or not question.strip():
                    raise SystemExit(
                        f"Missing question in {exclude_path}:{line_number}"
                    )
                excluded_questions.add(normalized_text(question))
                excluded_rows += 1
    seen_questions.update(excluded_questions)

    math_dataset = load_dataset(
        MATH_DATASET,
        "default",
        split="train",
        revision=args.math_revision,
    )
    math_candidates = []
    for index, source in enumerate(math_dataset):
        if source["level"] not in MATH_LEVEL_QUOTAS:
            rejections["math_level"] += 1
            continue
        answer = math_answer(source["solution"])
        if answer is None:
            rejections["math_non_integer_answer"] += 1
            continue
        row = build_sft_row(
            identifier=f"external-math-{index:05d}",
            question=source["problem"],
            solution=source["solution"],
            answer=answer,
            teacher="MATH-lighteval-human",
            metadata={
                "dataset": MATH_DATASET,
                "revision": args.math_revision,
                "split": "train",
                "level": source["level"],
                "type": source["type"],
            },
        )
        if filter_candidate(
            row,
            tokenizer,
            decontaminator,
            seen_questions,
            args.min_assistant_tokens,
            args.max_assistant_tokens,
            args.max_seq_length,
            rejections,
        ):
            math_candidates.append(row)
    selected_math = choose_by_quota(
        math_candidates, "level", math_level_quotas, args.seed, "math"
    )

    numina_dataset = load_dataset(
        NUMINA_DATASET,
        split="train",
        revision=args.numina_revision,
    )
    numina_candidates = []
    allowed_sources = set(numina_profile_quotas)
    for index, source in enumerate(numina_dataset):
        if source.get("source") not in allowed_sources:
            continue
        if source.get("synthetic") is True:
            rejections["numina_synthetic"] += 1
            continue
        if not valid_flag(source.get("problem_is_valid")) or not valid_flag(
            source.get("solution_is_valid")
        ):
            rejections["numina_invalid"] += 1
            continue
        if source.get("question_type") != "math-word-problem":
            rejections["numina_not_word_problem"] += 1
            continue
        answer = parse_integer(source.get("answer"))
        if answer is None:
            rejections["numina_non_integer_answer"] += 1
            continue
        row = build_sft_row(
            identifier=f"external-numina-{index:06d}",
            question=source["problem"],
            solution=source["solution"],
            answer=answer,
            teacher="NuminaMath-1.5-curated",
            metadata={
                "dataset": NUMINA_DATASET,
                "revision": args.numina_revision,
                "split": "train",
                "source": source["source"],
                "problem_type": source.get("problem_type"),
                "question_type": source.get("question_type"),
            },
        )
        if filter_candidate(
            row,
            tokenizer,
            decontaminator,
            seen_questions,
            args.min_assistant_tokens,
            args.max_assistant_tokens,
            args.max_seq_length,
            rejections,
        ):
            numina_candidates.append(row)
    selected_numina = choose_by_quota(
        numina_candidates, "source", numina_source_quotas, args.seed, "numina"
    )

    selected_gsm8k = []
    if args.gsm8k_count:
        gsm8k_dataset = load_dataset(
            GSM8K_DATASET,
            "main",
            split="train",
            revision=args.gsm8k_revision,
        )
        gsm8k_candidates = []
        for index, source in enumerate(gsm8k_dataset):
            parsed = gsm8k_solution(source["answer"])
            if parsed is None:
                rejections["gsm8k_non_integer_answer"] += 1
                continue
            solution, answer = parsed
            row = build_sft_row(
                identifier=f"external-gsm8k-{index:05d}",
                question=source["question"],
                solution=solution,
                answer=answer,
                teacher="GSM8K-human",
                metadata={
                    "dataset": GSM8K_DATASET,
                    "revision": args.gsm8k_revision,
                    "split": "train",
                },
            )
            if filter_candidate(
                row,
                tokenizer,
                decontaminator,
                seen_questions,
                args.min_assistant_tokens,
                args.max_assistant_tokens,
                args.max_seq_length,
                rejections,
            ):
                gsm8k_candidates.append(row)
        selected_gsm8k = sorted(
            gsm8k_candidates,
            key=lambda row: stable_score(args.seed, "gsm8k", row["question"]),
        )[: args.gsm8k_count]
        if len(selected_gsm8k) != args.gsm8k_count:
            raise SystemExit(
                f"Not enough GSM8K rows: {len(selected_gsm8k)} < {args.gsm8k_count}"
            )

    selected = selected_math + selected_numina + selected_gsm8k
    selected.sort(key=lambda row: row["id"])
    if len(selected) != args.math_count + args.numina_count + args.gsm8k_count:
        raise SystemExit(f"Unexpected selected count: {len(selected)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in selected:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    sequence_tokens = [row["metadata"]["sequence_tokens"] for row in selected]
    manifest = {
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "samples": len(selected),
        "seed": args.seed,
        "selection": "lowest sha256(seed:namespace:question) within fixed quotas",
        "official_decontamination": {
            "train_questions": len(official_train),
            "leaderboard_questions": len(official_leaderboard),
            "method": "normalized exact plus token-Jaccard/SequenceMatcher near duplicate",
        },
        "excluded_training_data": {
            "paths": [str(path) for path in args.exclude_data],
            "rows": excluded_rows,
            "unique_normalized_questions": len(excluded_questions),
        },
        "datasets": [
            {
                "id": GSM8K_DATASET,
                "revision": args.gsm8k_revision,
                "license": "MIT",
                "config": "main",
                "split": "train",
                "selected": len(selected_gsm8k),
            },
            {
                "id": MATH_DATASET,
                "revision": args.math_revision,
                "license": "MIT",
                "split": "train",
                "selected": len(selected_math),
            },
            {
                "id": NUMINA_DATASET,
                "revision": args.numina_revision,
                "license": "Apache-2.0",
                "split": "train",
                "selected": len(selected_numina),
            },
        ],
        "math_level_quotas": math_level_quotas,
        "math_type_counts": dict(
            sorted(Counter(row["metadata"]["type"] for row in selected_math).items())
        ),
        "numina_source_quotas": numina_source_quotas,
        "numina_profile": args.numina_profile,
        "numina_problem_type_counts": dict(
            sorted(
                Counter(
                    row["metadata"].get("problem_type") for row in selected_numina
                ).items()
            )
        ),
        "rejections": dict(sorted(rejections.items())),
        "token_limits": {
            "min_assistant_tokens": args.min_assistant_tokens,
            "max_assistant_tokens": args.max_assistant_tokens,
            "max_sequence_tokens": args.max_seq_length,
        },
        "selected_sequence_tokens": {
            "minimum": min(sequence_tokens),
            "average": sum(sequence_tokens) / len(sequence_tokens),
            "maximum": max(sequence_tokens),
        },
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
