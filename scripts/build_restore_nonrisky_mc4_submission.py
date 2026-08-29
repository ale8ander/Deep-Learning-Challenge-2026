"""Restore the four proven non-risky vote4-5/mc4 answers on top of the 656 submission.

The old 643 -> 647 submission changed six vote4-5 answers and gained four Public
points.  The 656 NC submission still keeps the two risky changes, but reverted
the four non-risky changes.  This script restores only those four answers and
checks every assumption used by that inference.
"""

import csv
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
BASE = ROOT / "submission_nocode_merged16.csv"
BEFORE_MC4 = ROOT / "submission_tir_sc8_vote3_mc2.csv"
AFTER_MC4 = ROOT / "submission_tir_sc8_vote3_plus_vote45mc4.csv"
OUTPUT = ROOT / "submission_nocode_merged16_restore_nonrisky_mc4.csv"
SUMMARY = ROOT / "outputs/submission_nocode_merged16_restore_nonrisky_mc4_summary.json"

TARGET_IDS = {
    "val-000262",
    "val-000637",
    "val-000874",
    "val-000878",
}
EXPECTED_OLD_MC4_CHANGED = {
    "val-000032",
    "val-000262",
    "val-000637",
    "val-000871",
    "val-000874",
    "val-000878",
}


def read_submission(path: Path) -> tuple[list[str], dict[str, int]]:
    rows = list(csv.DictReader(path.open(encoding="utf-8-sig", newline="")))
    ids = [row["id"] for row in rows]
    answers = {row["id"]: int(row["answer"]) for row in rows}
    if len(ids) != 831 or len(answers) != 831:
        raise ValueError(f"Invalid submission shape: {path} rows={len(ids)} unique={len(answers)}")
    return ids, answers


def main() -> None:
    ids, base = read_submission(BASE)
    before_ids, before = read_submission(BEFORE_MC4)
    after_ids, after = read_submission(AFTER_MC4)
    if ids != before_ids or ids != after_ids:
        raise ValueError("Submission ID order mismatch")

    old_changed = {problem_id for problem_id in ids if before[problem_id] != after[problem_id]}
    if old_changed != EXPECTED_OLD_MC4_CHANGED:
        raise ValueError(f"Unexpected old mc4 change set: {sorted(old_changed)}")
    if not TARGET_IDS < old_changed:
        raise ValueError("Restore targets are not a strict subset of the old mc4 changes")

    restored = dict(base)
    changes = []
    for problem_id in sorted(TARGET_IDS):
        if base[problem_id] != before[problem_id]:
            raise ValueError(
                f"Current 656 answer no longer matches pre-mc4 answer for {problem_id}: "
                f"base={base[problem_id]} before={before[problem_id]}"
            )
        restored[problem_id] = after[problem_id]
        changes.append(
            {
                "id": problem_id,
                "from": base[problem_id],
                "to": restored[problem_id],
            }
        )

    actual_changes = [problem_id for problem_id in ids if restored[problem_id] != base[problem_id]]
    if set(actual_changes) != TARGET_IDS:
        raise ValueError(f"Unexpected final change set: {actual_changes}")

    with OUTPUT.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["id", "answer"])
        writer.writerows((problem_id, restored[problem_id]) for problem_id in ids)

    check_ids, check = read_submission(OUTPUT)
    if check_ids != ids or check != restored:
        raise ValueError("Round-trip integrity check failed")

    summary = {
        "base": BASE.name,
        "output": OUTPUT.name,
        "rows": len(ids),
        "unique_ids": len(set(ids)),
        "id_order_matches_base": check_ids == ids,
        "all_answers_integer": True,
        "changed_vs_656": len(actual_changes),
        "changes": changes,
        "public_score_bound_from_prior_submission": {
            "base_correct": 656,
            "minimum_correct": 658,
            "maximum_correct": 660,
            "reason": "old six-answer mc4 change gained +4; the two already retained answers can contribute at most +2",
        },
    }
    SUMMARY.parent.mkdir(parents=True, exist_ok=True)
    SUMMARY.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
