"""Combine existing holdout464 prediction files into ensembles without new inference."""
import csv
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/holdout/official_holdout_464_clean.csv"

MODELS = {
    "hybrid3145": "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl",
    "grpo24": "outputs/grpo_3145_passrate94_steps24_holdout464_retry2048.jsonl",
    "grpo96": "outputs/grpo_3145_passrate94_steps96_holdout464_retry2048.jsonl",
    "verbose": "outputs/verbose_distill_holdout464_retry2048.jsonl",
}

SC_FILES = {
    "grpo96_n8": "outputs/self_consistency_grpo96_n8_holdout464_seed20260826.jsonl",
}


def norm(v):
    if v is None:
        return None
    s = str(v).strip()
    if s == "" or s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return s


def load_gold():
    gold = {}
    with open(GOLD, newline="") as f:
        for row in csv.DictReader(f):
            gold[row["id"]] = norm(row["answer"])
    return gold


def load_preds(path):
    out = {}
    with open(ROOT / path) as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = norm(d.get("prediction"))
    return out


def load_samples(path):
    out = {}
    with open(ROOT / path) as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = [norm(x) for x in d.get("sample_predictions", []) if norm(x) is not None]
    return out


def score(gold, pred):
    return sum(1 for i, a in gold.items() if pred.get(i) is not None and pred[i] == a)


def vote(ids, preds_list, tie_idx=0):
    """Plurality vote; ties broken by preds_list[tie_idx]."""
    out = {}
    for i in ids:
        vals = [p.get(i) for p in preds_list if p.get(i) is not None]
        if not vals:
            out[i] = None
            continue
        c = Counter(vals)
        top = c.most_common(1)[0][1]
        winners = [v for v, n in c.items() if n == top]
        if len(winners) == 1:
            out[i] = winners[0]
        else:
            tb = preds_list[tie_idx].get(i)
            out[i] = tb if tb in winners else winners[0]
    return out


def main():
    gold = load_gold()
    ids = list(gold)
    preds = {k: load_preds(v) for k, v in MODELS.items()}
    samples = {k: load_samples(v) for k, v in SC_FILES.items()}

    print(f"holdout464 gold: {len(gold)}\n")
    print("== 단독 모델 ==")
    for k, p in preds.items():
        print(f"  {k:12s} {score(gold, p)}/464")

    print("\n== 조합 oracle (하나라도 맞으면 정답) ==")
    names = list(preds)
    for r in range(2, len(names) + 1):
        for combo in combinations(names, r):
            oracle = sum(
                1 for i in ids if any(preds[n].get(i) == gold[i] for n in combo)
            )
            print(f"  {'+'.join(combo):40s} oracle {oracle}/464")

    print("\n== 조합 실채택 (plurality, tie=grpo96) ==")
    for r in range(2, len(names) + 1):
        for combo in combinations(names, r):
            order = sorted(combo, key=lambda n: 0 if n == "grpo96" else 1)
            v = vote(ids, [preds[n] for n in order], tie_idx=0)
            print(f"  {'+'.join(order):40s} vote   {score(gold, v)}/464")

    # GRPO96 N=8 samples + verbose deterministic
    sc = samples["grpo96_n8"]
    print("\n== GRPO96 N=8 self-consistency + verbose 결합 ==")
    base_oracle = sum(1 for i in ids if gold[i] in sc.get(i, []))
    print(f"  grpo96 N=8 sample oracle             {base_oracle}/464")
    exp = sum(
        1 for i in ids if gold[i] in sc.get(i, []) or preds["grpo96"].get(i) == gold[i]
    )
    print(f"  + grpo96 deterministic               {exp}/464")
    exp2 = sum(
        1
        for i in ids
        if gold[i] in sc.get(i, [])
        or preds["grpo96"].get(i) == gold[i]
        or preds["verbose"].get(i) == gold[i]
    )
    print(f"  + verbose deterministic              {exp2}/464")
    exp3 = sum(
        1
        for i in ids
        if gold[i] in sc.get(i, [])
        or any(preds[n].get(i) == gold[i] for n in names)
    )
    print(f"  + 모든 4모델 deterministic            {exp3}/464")

    print("\n== 실채택 규칙: grpo96 N=8 min_count + verbose override ==")
    for min_count in range(3, 9):
        picked = {}
        for i in ids:
            s = sc.get(i, [])
            c = Counter(s)
            det = preds["grpo96"].get(i)
            if c:
                val, n = c.most_common(1)[0]
                tops = [v for v, k in c.items() if k == n]
                picked[i] = val if (n >= min_count and len(tops) == 1) else det
            else:
                picked[i] = det
        base = score(gold, picked)

        # verbose as extra voter: use verbose answer when grpo96 sample support is weak
        for vthr in (2, 3):
            alt = dict(picked)
            for i in ids:
                c = Counter(sc.get(i, []))
                vb = preds["verbose"].get(i)
                if vb is None:
                    continue
                top = c.most_common(1)[0][1] if c else 0
                if top < vthr:
                    alt[i] = vb
            print(
                f"  min{min_count} = {base}/464 | verbose fallback(top<{vthr}) = {score(gold, alt)}/464"
            )

    print("\n== 3-voter: grpo96 N=8 plurality vs verbose vs grpo96 det ==")
    for min_count in range(4, 8):
        picked = {}
        for i in ids:
            c = Counter(sc.get(i, []))
            det = preds["grpo96"].get(i)
            vb = preds["verbose"].get(i)
            sc_pick = None
            if c:
                val, n = c.most_common(1)[0]
                tops = [v for v, k in c.items() if k == n]
                if n >= min_count and len(tops) == 1:
                    sc_pick = val
            cands = [x for x in (sc_pick, vb, det) if x is not None]
            cc = Counter(cands)
            if cc and cc.most_common(1)[0][1] >= 2:
                picked[i] = cc.most_common(1)[0][0]
            else:
                picked[i] = sc_pick if sc_pick is not None else det
        print(f"  min{min_count} 3-voter agree      {score(gold, picked)}/464")


if __name__ == "__main__":
    sys.exit(main())
