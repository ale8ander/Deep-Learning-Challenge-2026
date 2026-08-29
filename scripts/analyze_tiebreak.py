"""동률 tie-break 실험 — GPU 0, 기존 결과 재해석만.

발견: 놓친 22문제 중 8개가 표 차이 0(동률)이고, 그 중 3개는 정답이 공동 1위다.
현재 규칙은 동률이면 무조건 baseline으로 넘겨서 그 정답을 버린다.

여기서 시험하는 tie-break 후보:
  none      : 현재 규칙 (동률 -> baseline)
  baseline  : 동률 후보 중 baseline 답과 같은 게 있으면 그것
  scpool    : 기존 hybrid3145 N=8 SC 풀의 표가 많은 쪽
  lineage   : 계보 우선순위(hybrid3145 > grpo96 > verbose > tirsft) 상 먼저인 쪽
  smallest  : 절대값이 작은 답 (수학 대회 답은 대체로 작다는 사전지식)
  first     : Counter 순서(가장 먼저 등장) — 대조군
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tir_inference import normalize  # noqa: E402

SUBSET = ROOT / "data/holdout/holdout464_vote3.csv"
BASE = ROOT / "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl"
SC_POOL = ROOT / "outputs/self_consistency_hybrid3145_n8_holdout500.jsonl"
LINEAGE_ORDER = ["hybrid3145", "grpo96", "verbose", "tirsft"]
LINEAGES = {
    "hybrid3145": "outputs/tir_sc8_holdout464_vote3_to60.jsonl",
    "grpo96": "outputs/tir_sc8_holdout464_grpo96.jsonl",
    "verbose": "outputs/tir_sc8_holdout464_verbose.jsonl",
    "tirsft": "outputs/tir_sc8_holdout464_tirsft_fixed.jsonl",
}


def main():
    gold = {r["id"]: normalize(r["answer"])
            for r in csv.DictReader(open(SUBSET, encoding="utf-8-sig"))}
    base = {}
    for line in open(BASE):
        r = json.loads(line)
        if r["id"] in gold:
            base[r["id"]] = normalize(r.get("prediction"))
    scpool = {}
    for line in open(SC_POOL):
        r = json.loads(line)
        if r["id"] in gold:
            scpool[r["id"]] = Counter(
                x for x in (normalize(y) for y in r.get("sample_predictions", [])) if x is not None
            )

    pools = {}
    for name, path in LINEAGES.items():
        p = ROOT / path
        if not p.exists():
            continue
        d = {}
        for line in open(p):
            r = json.loads(line)
            d[r["id"]] = Counter({normalize(k): v for k, v in r["verified_counts"].items()
                                  if normalize(k) is not None})
        pools[name] = d

    ids = [i for i in sorted(gold) if i in base and all(i in d for d in pools.values())]
    b_all = sum(1 for i in ids if base[i] == gold[i])
    calib = [i for i in ids if int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) % 2 == 0]
    valid = [i for i in ids if i not in set(calib)]

    def tiebreak(kind, i, tied, combo):
        if kind == "none":
            return None
        if kind == "baseline":
            return base[i] if base[i] in tied else None
        if kind == "scpool":
            c = scpool.get(i, Counter())
            ranked = sorted(tied, key=lambda a: (-c[a], abs(a)))
            return ranked[0] if c[ranked[0]] > 0 else None
        if kind == "lineage":
            for n in LINEAGE_ORDER:
                if n not in combo:
                    continue
                for a in tied:
                    if pools[n][i].get(a, 0) > 0:
                        return a
            return None
        if kind == "smallest":
            return min(tied, key=lambda a: (abs(a), a))
        if kind == "first":
            return tied[0]
        return None

    print(f"대상 {len(ids)}문제  baseline {b_all}  calib {len(calib)} / valid {len(valid)}\n")
    print(f"{'조합':26s} {'mc':>2s} {'tie-break':10s} {'점수':>8s} {'차이':>5s} {'gain':>5s} {'reg':>4s} {'cal':>5s} {'val':>5s}")

    names = sorted(pools)
    rows = []
    for size in (1, 2, 3, 4):
        for combo in combinations(names, size):
            for mc in range(1, 3 * size + 2):
                for kind in ("none", "baseline", "scpool", "lineage", "smallest", "first"):
                    pred = {}
                    for i in ids:
                        c = Counter()
                        for n in combo:
                            c.update(pools[n][i])
                        top = c.most_common()
                        if not top or top[0][1] < mc:
                            pred[i] = base[i]
                            continue
                        best = top[0][1]
                        tied = [a for a, v in top if v == best]
                        if len(tied) == 1:
                            pred[i] = tied[0]
                        else:
                            choice = tiebreak(kind, i, tied, combo)
                            pred[i] = choice if choice is not None else base[i]
                    n_ok = sum(1 for i in ids if pred[i] == gold[i])
                    g = sum(1 for i in ids if base[i] != gold[i] and pred[i] == gold[i])
                    rg = sum(1 for i in ids if base[i] == gold[i] and pred[i] != gold[i])
                    nc = sum(1 for i in calib if pred[i] == gold[i]) - sum(1 for i in calib if base[i] == gold[i])
                    nv = sum(1 for i in valid if pred[i] == gold[i]) - sum(1 for i in valid if base[i] == gold[i])
                    rows.append((n_ok, "+".join(combo), mc, kind, g, rg, nc, nv))

    rows.sort(reverse=True)
    seen = set()
    for n_ok, combo, mc, kind, g, rg, nc, nv in rows[:25]:
        key = (combo, mc, kind)
        if key in seen:
            continue
        seen.add(key)
        print(f"{combo:26s} {mc:2d} {kind:10s} {n_ok:4d}/{len(ids)} {n_ok-b_all:+5d} {g:5d} {rg:4d} {nc:+5d} {nv:+5d}")

    print()
    best = rows[0]
    print(f"최고: {best[1]} mc={best[2]} tie-break={best[3]} -> {best[0]}/{len(ids)} ({best[0]-b_all:+d}), "
          f"gain {best[4]} reg {best[5]}, calib {best[6]:+d} valid {best[7]:+d}")
    print("참고 — 현재 제출 규칙(hybrid3145 mc=2, tie-break 없음): +13")


if __name__ == "__main__":
    main()
