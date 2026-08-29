import argparse
from collections import Counter
import csv
import hashlib
import json
from pathlib import Path

from datasets import load_dataset
from transformers import AutoTokenizer

from prepare_hard_math_sft import (
    MATH_DATASET,
    MATH_REVISION,
    NUMINA_DATASET,
    NUMINA_REVISION,
    OfficialDecontaminator,
    VISUAL_PATTERN,
    math_answer,
    ngrams,
    normalized_text,
    parse_integer,
    read_csv,
    stable_score,
    valid_flag,
)


MATH_TYPES = ("Number Theory", "Counting & Probability")
MATH_LEVELS = ("Level 4", "Level 5")
NUMINA_DIRECT_SOURCE = "number_theory"
NUMINA_OLYMPIAD_SOURCES = {
    "olympiads",
    "aops_forum",
    "cn_contest",
    "amc_aime",
}
NUMBER_THEORY_KEYWORDS = (
    "number theory",
    "mod",
    "modulo",
    "congruent",
    "congruence",
    "gcd",
    "lcm",
    "prime",
    "primes",
    "divisible",
    "divisibility",
    "divisor",
    "divisors",
    "remainder",
    "coprime",
    "diophantine",
    "residue",
    "factorization",
    "perfect square",
)
COMBINATORICS_KEYWORDS = (
    "combinatorics",
    "counting",
    "permutation",
    "permutations",
    "combination",
    "combinations",
    "arrangement",
    "arrangements",
    "number of ways",
    "how many ways",
    "binomial",
    "pigeonhole",
    "inclusion exclusion",
    "subset",
    "subsets",
    "coloring",
    "probability",
)


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read_jsonl(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as file:
        return [json.loads(line) for line in file if line.strip()]


def path_rows_and_questions(path: Path) -> tuple[int, list[str]]:
    if path.suffix.casefold() == ".csv":
        rows = read_csv(path)
    elif path.suffix.casefold() == ".jsonl":
        rows = read_jsonl(path)
    else:
        raise SystemExit(f"Unsupported exclusion file: {path}")
    questions = []
    for row in rows:
        question = row.get("question")
        if not isinstance(question, str):
            messages = row.get("messages")
            if isinstance(messages, list):
                for message in messages:
                    if message.get("role") == "user" and isinstance(
                        message.get("content"), str
                    ):
                        question = message["content"]
                        break
        if isinstance(question, str) and question.strip():
            questions.append(question.strip())
    return len(rows), questions


class IncrementalNearDuplicateIndex:
    def __init__(self) -> None:
        self.questions: list[str] = []
        self.normalized: list[str] = []
        self.tokens: list[list[str]] = []
        self.exact: set[str] = set()
        self.index: dict[tuple[str, ...], set[int]] = {}

    def add(self, question: str) -> None:
        normalized = normalized_text(question)
        tokens = normalized.split()
        position = len(self.normalized)
        self.questions.append(question)
        self.normalized.append(normalized)
        self.tokens.append(tokens)
        self.exact.add(normalized)
        for gram in ngrams(tokens):
            self.index.setdefault(gram, set()).add(position)

    def match(self, question: str) -> str | None:
        from difflib import SequenceMatcher

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
            other_tokens = self.tokens[index]
            other_set = set(other_tokens)
            union = token_set | other_set
            jaccard = len(token_set & other_set) / len(union) if union else 1.0
            length_ratio = min(len(tokens), len(other_tokens)) / max(
                len(tokens), len(other_tokens)
            )
            if jaccard >= 0.85 and length_ratio >= 0.80:
                return "near_token"
            if length_ratio >= 0.85 and SequenceMatcher(
                None, normalized, self.normalized[index], autojunk=False
            ).ratio() >= 0.92:
                return "near_sequence"
        return None


def keyword_domain(question: str, problem_type: object) -> tuple[str | None, list[str]]:
    normalized = normalized_text(question)
    nt_hits = [term for term in NUMBER_THEORY_KEYWORDS if term in normalized]
    comb_hits = [term for term in COMBINATORICS_KEYWORDS if term in normalized]
    kind = str(problem_type or "").casefold()
    if nt_hits and not comb_hits:
        return "number_theory", nt_hits
    if comb_hits and not nt_hits:
        return "combinatorics", comb_hits
    if nt_hits and comb_hits:
        if "number theory" in kind:
            return "number_theory", nt_hits + comb_hits
        if "combinator" in kind:
            return "combinatorics", nt_hits + comb_hits
        if len(nt_hits) >= len(comb_hits):
            return "number_theory", nt_hits + comb_hits
        return "combinatorics", nt_hits + comb_hits
    return None, []


def choose(
    candidates: list[dict],
    count: int,
    seed: int,
    namespace: str,
    selected_index: IncrementalNearDuplicateIndex,
    rejections: Counter,
) -> list[dict]:
    ranked = sorted(
        candidates,
        key=lambda row: stable_score(seed, namespace, row["question"]),
    )
    selected = []
    for row in ranked:
        duplicate = selected_index.match(row["question"])
        if duplicate:
            rejections[f"selected_internal_{duplicate}"] += 1
            continue
        selected.append(row)
        selected_index.add(row["question"])
        if len(selected) == count:
            break
    if len(selected) != count:
        raise SystemExit(
            f"Not enough rows for {namespace}: selected={len(selected)} "
            f"requested={count} candidates={len(candidates)}"
        )
    return selected


def make_row(
    identifier: str,
    question: str,
    answer: str,
    domain: str,
    dataset: str,
    revision: str,
    metadata: dict,
) -> dict:
    return {
        "id": identifier,
        "question": question.strip(),
        "answer": answer,
        "domain": domain,
        "dataset": dataset,
        "revision": revision,
        "metadata": metadata,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data/processed/rigor_ntcomb_candidates_500.jsonl"),
    )
    parser.add_argument("--target-count", type=int, default=500)
    parser.add_argument("--math-count", type=int, default=250)
    parser.add_argument("--numina-count", type=int, default=250)
    parser.add_argument("--seed", type=int, default=20260826)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument(
        "--model-path",
        type=Path,
        default=Path("/workspace/models/Qwen2.5-3B-Instruct"),
    )
    parser.add_argument("--max-question-tokens", type=int, default=768)
    parser.add_argument("--exclude-path", type=Path, action="append", default=[])
    parser.add_argument("--math-revision", default=MATH_REVISION)
    parser.add_argument("--numina-revision", default=NUMINA_REVISION)
    args = parser.parse_args()

    if args.target_count != args.math_count + args.numina_count:
        raise SystemExit("--target-count must equal --math-count + --numina-count")
    if args.math_count % 2 or args.numina_count % 2:
        raise SystemExit("Math and Numina counts must be even for balanced domains")

    official_paths = [
        args.data_dir / "deep_chal_math_train.csv",
        args.data_dir / "deep_chal_math_leaderboard_filtered.csv",
    ]
    all_exclusion_paths = official_paths + args.exclude_path
    missing = [str(path) for path in all_exclusion_paths if not path.is_file()]
    if missing:
        raise SystemExit(f"Missing required exclusion files: {missing}")

    exclusion_records = []
    exclusion_questions = []
    normalized_sources: dict[str, set[str]] = {}
    for path in all_exclusion_paths:
        rows, questions = path_rows_and_questions(path)
        normalized = {normalized_text(question) for question in questions}
        normalized_sources[str(path)] = normalized
        exclusion_questions.extend(questions)
        exclusion_records.append(
            {
                "path": str(path),
                "sha256": sha256(path),
                "rows": rows,
                "questions": len(questions),
                "unique_normalized_questions": len(normalized),
            }
        )
    unique_exclusion_questions = {
        normalized_text(question): question for question in exclusion_questions
    }
    decontaminator = OfficialDecontaminator(
        list(unique_exclusion_questions.values())
    )
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_path, local_files_only=True
    )
    rejections: Counter = Counter()

    math_dataset = load_dataset(
        MATH_DATASET,
        "default",
        split="train",
        revision=args.math_revision,
    )
    math_candidates: dict[str, list[dict]] = {
        "number_theory": [],
        "combinatorics": [],
    }
    for index, source in enumerate(math_dataset):
        if source.get("type") not in MATH_TYPES:
            rejections["math_type"] += 1
            continue
        if source.get("level") not in MATH_LEVELS:
            rejections["math_level"] += 1
            continue
        answer = math_answer(source.get("solution") or "")
        if answer is None:
            rejections["math_non_integer_answer"] += 1
            continue
        question = str(source.get("problem") or "").strip()
        if not question:
            rejections["math_empty"] += 1
            continue
        if VISUAL_PATTERN.search(question):
            rejections["math_visual_or_url"] += 1
            continue
        duplicate = decontaminator.match(question)
        if duplicate:
            rejections[f"math_exclusion_{duplicate}"] += 1
            continue
        question_tokens = len(tokenizer.encode(question, add_special_tokens=False))
        if question_tokens > args.max_question_tokens:
            rejections["math_question_too_long"] += 1
            continue
        domain = (
            "number_theory"
            if source["type"] == "Number Theory"
            else "combinatorics"
        )
        math_candidates[domain].append(
            make_row(
                f"rigor-math-{index:05d}",
                question,
                answer,
                domain,
                MATH_DATASET,
                args.math_revision,
                {
                    "split": "train",
                    "level": source["level"],
                    "type": source["type"],
                    "question_tokens": question_tokens,
                },
            )
        )

    numina_dataset = load_dataset(
        NUMINA_DATASET,
        split="train",
        revision=args.numina_revision,
    )
    numina_candidates: dict[str, list[dict]] = {
        "number_theory": [],
        "combinatorics": [],
    }
    for index, source in enumerate(numina_dataset):
        source_name = source.get("source")
        if source_name != NUMINA_DIRECT_SOURCE and source_name not in NUMINA_OLYMPIAD_SOURCES:
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
        question = str(source.get("problem") or "").strip()
        if not question:
            rejections["numina_empty"] += 1
            continue
        if source_name == NUMINA_DIRECT_SOURCE:
            domain = "number_theory"
            hits = ["source:number_theory"]
        else:
            domain, hits = keyword_domain(question, source.get("problem_type"))
            if domain is None:
                rejections["numina_no_ntcomb_keyword"] += 1
                continue
        if VISUAL_PATTERN.search(question):
            rejections["numina_visual_or_url"] += 1
            continue
        duplicate = decontaminator.match(question)
        if duplicate:
            rejections[f"numina_exclusion_{duplicate}"] += 1
            continue
        question_tokens = len(tokenizer.encode(question, add_special_tokens=False))
        if question_tokens > args.max_question_tokens:
            rejections["numina_question_too_long"] += 1
            continue
        numina_candidates[domain].append(
            make_row(
                f"rigor-numina-{index:06d}",
                question,
                answer,
                domain,
                NUMINA_DATASET,
                args.numina_revision,
                {
                    "split": "train",
                    "source": source_name,
                    "problem_type": source.get("problem_type"),
                    "question_type": source.get("question_type"),
                    "keyword_hits": sorted(set(hits)),
                    "question_tokens": question_tokens,
                },
            )
        )

    selected_index = IncrementalNearDuplicateIndex()
    selected = []
    math_per_domain = args.math_count // 2
    numina_per_domain = args.numina_count // 2
    for domain in ("number_theory", "combinatorics"):
        selected.extend(
            choose(
                math_candidates[domain],
                math_per_domain,
                args.seed,
                f"math:{domain}",
                selected_index,
                rejections,
            )
        )
    for domain in ("number_theory", "combinatorics"):
        selected.extend(
            choose(
                numina_candidates[domain],
                numina_per_domain,
                args.seed,
                f"numina:{domain}",
                selected_index,
                rejections,
            )
        )
    selected.sort(key=lambda row: row["id"])
    if len(selected) != args.target_count:
        raise SystemExit(f"Unexpected selected count: {len(selected)}")
    if len({row["id"] for row in selected}) != len(selected):
        raise SystemExit("Duplicate selected IDs")

    # Final independent exact audit against every requested exclusion source.
    exact_overlap_by_path = {}
    for path, normalized in normalized_sources.items():
        overlaps = sum(normalized_text(row["question"]) in normalized for row in selected)
        exact_overlap_by_path[path] = overlaps
        if overlaps:
            raise SystemExit(f"Exact overlap remained against {path}: {overlaps}")
    near_overlap = Counter()
    for row in selected:
        kind = decontaminator.match(row["question"])
        if kind:
            near_overlap[kind] += 1
    if near_overlap:
        raise SystemExit(f"Near overlap remained: {dict(near_overlap)}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in selected:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")

    manifest = {
        "output": str(args.output),
        "output_sha256": sha256(args.output),
        "samples": len(selected),
        "seed": args.seed,
        "selection": "balanced deterministic sha256 ranking after target filters and exact/near decontamination",
        "target": {
            "math_count": args.math_count,
            "numina_count": args.numina_count,
            "domains": {
                key: value
                for key, value in sorted(
                    Counter(row["domain"] for row in selected).items()
                )
            },
        },
        "datasets": [
            {
                "id": MATH_DATASET,
                "revision": args.math_revision,
                "license": "MIT",
                "split": "train",
                "filters": {
                    "type": list(MATH_TYPES),
                    "level": list(MATH_LEVELS),
                    "integer_answer": True,
                },
                "selected": sum(row["dataset"] == MATH_DATASET for row in selected),
            },
            {
                "id": NUMINA_DATASET,
                "revision": args.numina_revision,
                "license": "Apache-2.0",
                "split": "train",
                "filters": {
                    "direct_source": NUMINA_DIRECT_SOURCE,
                    "olympiad_sources": sorted(NUMINA_OLYMPIAD_SOURCES),
                    "olympiad_keyword_filter": True,
                    "integer_answer": True,
                    "valid_problem_and_solution": True,
                    "non_synthetic": True,
                },
                "selected": sum(row["dataset"] == NUMINA_DATASET for row in selected),
            },
        ],
        "selected_source_counts": dict(
            sorted(
                Counter(
                    row["metadata"].get("source", row["metadata"].get("type"))
                    for row in selected
                ).items()
            )
        ),
        "candidate_counts_before_quota": {
            "math": {key: len(value) for key, value in math_candidates.items()},
            "numina": {key: len(value) for key, value in numina_candidates.items()},
        },
        "decontamination": {
            "method": "normalized exact plus token-Jaccard/SequenceMatcher near duplicate",
            "exclusion_files": exclusion_records,
            "unique_normalized_exclusion_questions": len(unique_exclusion_questions),
            "selected_exact_overlap_by_path": exact_overlap_by_path,
            "selected_near_overlap_total": sum(near_overlap.values()),
            "selected_internal_exact_or_near_duplicates": 0,
        },
        "rejections": dict(sorted(rejections.items())),
        "max_question_tokens": args.max_question_tokens,
    }
    manifest_path = args.output.with_suffix(".manifest.json")
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
