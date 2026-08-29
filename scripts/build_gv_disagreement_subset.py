"""Step 2 준비 — GRPO96 deterministic과 verbose distill deterministic의 답이 갈린 문제만 뽑는다.

두 계보가 합의한 413문제는 84.5% 정확도라 건드리지 않고,
답이 갈린 51문제에만 추가 stochastic 샘플을 투입하기 위한 서브셋이다.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/holdout/official_holdout_464_clean.csv"
GRPO = ROOT / "outputs/grpo_3145_passrate94_steps96_holdout464_retry2048.jsonl"
VERBOSE = ROOT / "outputs/verbose_distill_holdout464_retry2048.jsonl"
OUT_Q = ROOT / "data/holdout/holdout464_gv_disagree51.csv"
OUT_P = ROOT / "outputs/gv_disagree51_grpo96_det_predictions.csv"


def norm(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


def load(path):
    out = {}
    for line in open(path):
        r = json.loads(line)
        out[r["id"]] = norm(r.get("prediction"))
    return out


def main():
    with open(GOLD, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    g, v = load(GRPO), load(VERBOSE)

    picked = [r for r in rows if g.get(r["id"]) is None or g[r["id"]] != v.get(r["id"])]

    with open(OUT_Q, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "answer"])
        w.writeheader()
        for r in picked:
            w.writerow({"id": r["id"], "question": r["question"], "answer": r["answer"]})

    with open(OUT_P, "w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "prediction"])
        w.writeheader()
        for r in picked:
            w.writerow({"id": r["id"], "prediction": g.get(r["id"])})

    correct = sum(1 for r in picked if norm(r["answer"]) == g.get(r["id"]))
    print(f"불일치 문제: {len(picked)}개 -> {OUT_Q.name}, {OUT_P.name}")
    print(f"  이 중 GRPO96 deterministic이 맞은 문제: {correct}")


if __name__ == "__main__":
    main()
