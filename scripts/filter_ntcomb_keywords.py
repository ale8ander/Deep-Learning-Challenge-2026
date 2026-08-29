import argparse
from collections import Counter
import csv
import json
import re
from pathlib import Path


NUMBER_THEORY_PATTERNS = {
    "modular": r"\\pmod|\bmod(?:ulo)?\b|congruen(?:t|ce)",
    "gcd_lcm_coprime": r"\bgcd\b|\blcm\b|co-?prime|relatively prime",
    "prime": r"\bprime(?:s| factor| number)?\b",
    "divisibility": r"divisib|\bdivisor(?:s)?\b|number of (?:positive )?(?:integral )?divisors",
    "remainder": r"\bremainder(?:s)?\b|obtained by dividing",
    "multiple": r"\bmultiple of\b",
    "base_representation": r"\bbase[- ]?(?:[2-9]|1[0-9]|20)\b|binary representation|base of numeration",
    "digit_arithmetic": r"sum of (?:the )?digits|product of (?:the |its )?digits|cross sum|digit sum|digits? (?:is|are|forming|whose)",
    "integer_equation": r"(?:positive|nonnegative|natural|non-zero|nonzero) integers? satisfying|integer solutions?|diophantine",
    "perfect_power_integer": r"perfect (?:square|cube|power)",
    "palindrome_integer": r"\bpalindrome\b",
    "arithmetic_function": r"euler.?s? totient|arithmetic function|sum-of-divisors|number of divisors",
    "fraction_extremal": r"numerator and denominator|simple fractions",
}
COMBINATORICS_PATTERNS = {
    "permutation_combination": r"permut(?:ation|ations|e|ed)|combin(?:ation|ations|atorial|atorics)|\\binom",
    "subset_selection": r"\bsubsets?\b|[kKnN]-element subset|elements can be selected|select(?:ed|ing)? .* at most",
    "counting_phrase": r"\bcount the number\b|\bnumber of ways\b|\bhow many (?:ways|orders|arrangements|permutations|subsets|sequences|configurations|numbers|integers|positive integers|quadruples|triples)\b",
    "arrangement_order": r"\barrangements?\b|in how many orders|ordered pairs?|unordered pairs?",
    "distinguishable_distribution": r"distinguishable|indistinguishable|distribut(?:e|ed|ion) .* (?:balls|items|objects|boxes)",
    "probability": r"\bprobability\b|randomly (?:chosen|selected|arranged)|expected number",
    "graph_tournament": r"\bgraph\b|vertices and edges|football tournament|round-robin|single-round tournament",
    "extremal_configuration": r"maximum possible number|greatest number of elements|at most, such that|largest possible (?:set|collection|family)",
    "pigeonhole_inclusion": r"pigeonhole|inclusion-exclusion",
    "coloring_tiling_path": r"\bcolorings?\b|\btilings?\b|lattice paths?|grid paths?",
    "voting_assignment": r"\bvoting\b|votes? on the distribution|who should receive",
    "game_state_count": r"playing a game|after several rounds|orders can .* (?:eat|remove)",
    "integer_tuple_count": r"number of (?:ordered )?(?:pairs|triples|quadruples|tuples).*integers|quadruples? .* satisfying",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as file:
        return list(csv.DictReader(file))


def matches(text: str, patterns: dict[str, str]) -> list[str]:
    return [
        name
        for name, pattern in patterns.items()
        if re.search(pattern, text, flags=re.IGNORECASE | re.DOTALL)
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--id-subset-jsonl", type=Path)
    args = parser.parse_args()

    rows = read_csv(args.input)
    if args.id_subset_jsonl:
        with args.id_subset_jsonl.open(encoding="utf-8") as file:
            keep = {json.loads(line)["id"] for line in file if line.strip()}
        rows = [row for row in rows if row["id"] in keep]

    selected = []
    for row in rows:
        nt = matches(row["question"], NUMBER_THEORY_PATTERNS)
        comb = matches(row["question"], COMBINATORICS_PATTERNS)
        if not nt and not comb:
            continue
        category = "both" if nt and comb else (
            "number_theory" if nt else "combinatorics"
        )
        selected.append(
            {
                "id": row["id"],
                "question": row["question"],
                "answer": row["answer"],
                "keyword_category": category,
                "number_theory_matches": nt,
                "combinatorics_matches": comb,
            }
        )

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as file:
        for row in selected:
            file.write(json.dumps(row, ensure_ascii=False) + "\n")
    counts = Counter(row["keyword_category"] for row in selected)
    print(
        json.dumps(
            {
                "input_rows": len(rows),
                "selected": len(selected),
                "category_counts": dict(sorted(counts.items())),
                "number_theory_including_both": sum(
                    row["keyword_category"] in {"number_theory", "both"}
                    for row in selected
                ),
                "combinatorics_including_both": sum(
                    row["keyword_category"] in {"combinatorics", "both"}
                    for row in selected
                ),
                "output": str(args.output),
            },
            ensure_ascii=False,
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
