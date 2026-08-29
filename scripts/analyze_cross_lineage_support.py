"""합산 표수가 같아도 '두 계보 모두에 등장한 답'이 더 정확한가?

배경: 교차계보 plurality 규칙은 368/464에서 상한에 도달했는데(analyze_cross_lineage_agreement.py),
샘플 풀 oracle은 402/464다. 남은 34문제는 plurality가 아예 못 잡는 곳에 있다.
merged count는 같지만 계보 분포가 다른 후보(4+0 vs 2+2)를 구분하면
학습 없이 그 격차를 일부 회수할 수 있는지 검사한다.
"""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/holdout/official_holdout_464_clean.csv"
POOLS = {
    "grpo96": "outputs/self_consistency_grpo96_n8_holdout464_seed20260826.jsonl",
    "h3145": "outputs/self_consistency_hybrid3145_n8_holdout500.jsonl",
    "verbose": "outputs/self_consistency_verbose_n8_holdout464_seed20260827.jsonl",
}
DET_GRPO = "outputs/grpo_3145_passrate94_steps96_holdout464_retry2048.jsonl"


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


def main():
    gold = load_gold()
    pools = {k: v for k, v in ((k, load_pool(p)) for k, p in POOLS.items()) if v}
    names = [n for n in ("grpo96", "h3145") if n in pools]
    if len(names) < 2:
        print("두 계보 필요")
        return
    a, b = names
    ids = [i for i in gold if pools[a].get(i) and pools[b].get(i)]
    print(f"대상 {len(ids)}문제 ({a} + {b})\n")

    det = {}
    with open(ROOT / DET_GRPO) as f:
        for line in f:
            d = json.loads(line)
            det[d["id"]] = norm(d.get("prediction"))

    # (merged_count, lineage_breadth) 별 정답률
    stats = defaultdict(lambda: [0, 0])  # key -> [correct, total]
    for i in ids:
        ca, cb = Counter(pools[a][i]), Counter(pools[b][i])
        for val in set(ca) | set(cb):
            m = ca[val] + cb[val]
            breadth = (1 if ca[val] else 0) + (1 if cb[val] else 0)
            s = stats[(m, breadth)]
            s[1] += 1
            s[0] += val == gold[i]

    print("== 후보 단위: (합산표수, 등장계보수) 별 정답률 ==")
    print(f"{'표수':>4} {'계보':>4} {'정답/후보':>12} {'정답률':>8}")
    for (m, br) in sorted(stats):
        c, t = stats[(m, br)]
        if t >= 5:
            print(f"{m:>4} {br:>4} {c:>5}/{t:<6} {c/t:>8.1%}")

    print("\n== 같은 합산표수에서 계보 1개 vs 2개 직접 비교 ==")
    for m in range(2, 13):
        one = stats.get((m, 1), [0, 0])
        two = stats.get((m, 2), [0, 0])
        if one[1] >= 5 and two[1] >= 5:
            print(
                f"  {m:2d}표: 1계보 {one[0]}/{one[1]} = {one[0]/one[1]:.1%}   "
                f"2계보 {two[0]}/{two[1]} = {two[0]/two[1]:.1%}   "
                f"차이 {two[0]/two[1] - one[0]/one[1]:+.1%}"
            )

    print("\n== 선택 규칙: breadth 우선 정렬 ==")
    # 후보를 (breadth, count) 사전순으로 정렬해 1등을 고른다 vs (count, breadth)
    for mode, key in (
        ("count 우선 (기존)", lambda m, br: (m, br)),
        ("breadth 우선 (신규)", lambda m, br: (br, m)),
    ):
        for min_m in range(4, 10):
            hit = 0
            for i in ids:
                ca, cb = Counter(pools[a][i]), Counter(pools[b][i])
                cands = []
                for val in set(ca) | set(cb):
                    m = ca[val] + cb[val]
                    br = (1 if ca[val] else 0) + (1 if cb[val] else 0)
                    cands.append((key(m, br), m, val))
                cands.sort(reverse=True)
                best = cands[0]
                tie = sum(1 for c in cands if c[0] == best[0])
                hit += (best[2] if (tie == 1 and best[1] >= min_m) else det.get(i)) == gold[i]
            print(f"  {mode:22s} min합산{min_m} → {hit}/{len(ids)}")


if __name__ == "__main__":
    main()
