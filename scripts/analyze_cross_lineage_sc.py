"""Combine self-consistency sample pools from different model lineages on holdout464."""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/holdout/official_holdout_464_clean.csv"

POOLS = {
    "grpo96": "outputs/self_consistency_grpo96_n8_holdout464_seed20260826.jsonl",
    "h3145": "outputs/self_consistency_hybrid3145_n8_holdout500.jsonl",
    "verbose": "outputs/self_consistency_verbose_n8_holdout464_seed20260827.jsonl",
}
DET = {
    "grpo96": "outputs/grpo_3145_passrate94_steps96_holdout464_retry2048.jsonl",
    "h3145": "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl",
    "verbose": "outputs/verbose_distill_holdout464_retry2048.jsonl",
}


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


def load_gold():
    with open(GOLD, newline="") as f:
        return {r["id"]: norm(r["answer"]) for r in csv.DictReader(f)}


def load_pool(path):
    p = ROOT / path
    if not p.exists():
        return None
    out = {}
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = [x for x in (norm(v) for v in d.get("sample_predictions", [])) if x is not None]
    return out


def load_det(path):
    out = {}
    with open(ROOT / path) as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = norm(d.get("prediction"))
    return out


def evaluate(gold, ids, pools, dets, tie_model, min_count):
    hit = 0
    for i in ids:
        samples = []
        for p in pools:
            samples.extend(p.get(i, []))
        c = Counter(samples)
        pick = None
        if c:
            val, n = c.most_common(1)[0]
            tops = [v for v, k in c.items() if k == n]
            if n >= min_count and len(tops) == 1:
                pick = val
        if pick is None:
            pick = dets[tie_model].get(i)
        if pick == gold[i]:
            hit += 1
    return hit


def main():
    gold = load_gold()
    ids = list(gold)
    pools = {k: load_pool(v) for k, v in POOLS.items()}
    dets = {k: load_det(v) for k, v in DET.items()}
    avail = {k: v for k, v in pools.items() if v is not None}
    print("사용 가능한 샘플 풀:", ", ".join(avail))

    for name, pool in avail.items():
        cov = sum(1 for i in ids if pool.get(i))
        orac = sum(1 for i in ids if gold[i] in pool.get(i, []))
        print(f"  {name:8s} 커버 {cov}/464  sample oracle {orac}/464")

    print("\n== 단일 풀 min_count 스윕 (tie=자기 deterministic) ==")
    for name, pool in avail.items():
        row = [f"{m}:{evaluate(gold, ids, [pool], dets, name, m)}" for m in range(3, 9)]
        print(f"  {name:8s} " + "  ".join(row))

    names = sorted(avail)
    if len(names) >= 2:
        print("\n== 교차계보 결합 (샘플 합치기) ==")
        from itertools import combinations

        for r in range(2, len(names) + 1):
            for combo in combinations(names, r):
                ps = [avail[n] for n in combo]
                cov = sum(1 for i in ids if all(avail[n].get(i) for n in combo))
                orac = sum(1 for i in ids if any(gold[i] in avail[n].get(i, []) for n in combo))
                tie = "grpo96" if "grpo96" in combo else combo[0]
                row = [f"{m}:{evaluate(gold, ids, ps, dets, tie, m)}" for m in range(4, 13)]
                print(f"  {'+'.join(combo):24s} cov {cov:3d} oracle {orac}/464  " + "  ".join(row))


if __name__ == "__main__":
    main()
