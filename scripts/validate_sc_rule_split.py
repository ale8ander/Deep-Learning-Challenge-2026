"""min_count threshold 과적합 검증.

holdout464 전체에서 threshold를 고르면 12개 후보 x 여러 풀 조합을 같은 셋에서
비교하는 것이라 다중비교 과적합이 생긴다. 과거 fixed200에서 +-2 차이로
채택/기각을 반복했던 실패와 같은 구조다.

여기서는 id 해시로 464를 calibration/validation 절반씩 나누고,
calibration에서 고른 threshold를 validation에서만 채점한다.
"""
import csv
import hashlib
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
DETS = {
    "grpo96": "outputs/grpo_3145_passrate94_steps96_holdout464_retry2048.jsonl",
    "h3145": "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl",
    "verbose": "outputs/verbose_distill_holdout464_retry2048.jsonl",
}
COMBOS = [
    ("grpo96",),
    ("verbose",),
    ("grpo96", "h3145"),
    ("grpo96", "verbose"),
    ("h3145", "verbose"),
    ("grpo96", "h3145", "verbose"),
]


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


def main():
    with open(GOLD, newline="") as f:
        gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(f)}
    pools, dets = {}, {}
    for k, p in POOLS.items():
        pools[k] = {}
        with open(ROOT / p) as f:
            for line in f:
                d = json.loads(line)
                pools[k][d["id"]] = [x for x in (norm(v) for v in d.get("sample_predictions", [])) if x is not None]
    for k, p in DETS.items():
        dets[k] = {}
        with open(ROOT / p) as f:
            for line in f:
                d = json.loads(line)
                dets[k][d["id"]] = norm(d.get("prediction"))

    calib, valid = [], []
    for i in gold:
        h = int(hashlib.sha256(i.encode()).hexdigest(), 16)
        (calib if h % 2 == 0 else valid).append(i)
    print(f"calibration {len(calib)}문제 / validation {len(valid)}문제\n")

    base = dets["grpo96"]
    print(f"기준선 (GRPO96 deterministic): calib {sum(1 for i in calib if base[i]==gold[i])}/{len(calib)}  "
          f"valid {sum(1 for i in valid if base[i]==gold[i])}/{len(valid)}")
    h = dets["h3145"]
    print(f"기준선 (hybrid3145 baseline):  calib {sum(1 for i in calib if h[i]==gold[i])}/{len(calib)}  "
          f"valid {sum(1 for i in valid if h[i]==gold[i])}/{len(valid)}\n")

    def score(ids, combo, mc, tie):
        hit = 0
        for i in ids:
            s = []
            for n in combo:
                s.extend(pools[n].get(i, []))
            c = Counter(s)
            pick = None
            if c:
                val, k = c.most_common(1)[0]
                tops = [v for v, q in c.items() if q == k]
                if k >= mc and len(tops) == 1:
                    pick = val
            if pick is None:
                pick = dets[tie].get(i)
            hit += pick == gold[i]
        return hit

    print(f"{'조합':26s} {'tie':8s} {'best_mc':>7} {'calib':>10} {'valid':>10} {'전체':>9}")
    results = []
    for combo in COMBOS:
        tie = "grpo96" if "grpo96" in combo else combo[0]
        n = len(combo) * 8
        best_mc, best_c = None, -1
        for mc in range(3, n + 1):
            c = score(calib, combo, mc, tie)
            if c > best_c:
                best_c, best_mc = c, mc
        v = score(valid, combo, best_mc, tie)
        allsc = score(list(gold), combo, best_mc, tie)
        results.append((combo, tie, best_mc, best_c, v, allsc))
        print(
            f"{'+'.join(combo):26s} {tie:8s} {best_mc:>7} "
            f"{best_c:>4}/{len(calib):<5} {v:>4}/{len(valid):<5} {allsc:>4}/464"
        )

    print("\n== validation 기준 순위 (calibration에서 고른 threshold를 그대로 적용) ==")
    for combo, tie, mc, c, v, a in sorted(results, key=lambda r: -r[4]):
        base_v = sum(1 for i in valid if dets["h3145"][i] == gold[i])
        print(f"  {'+'.join(combo):26s} mc={mc:<3} validation {v}/{len(valid)}  (hybrid3145 {base_v}, {v-base_v:+d})")


if __name__ == "__main__":
    main()
