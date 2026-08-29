"""Step 1 후속 — 합의 게이트 방식의 헤드룸을 정확히 잰다.

핵심 질문 2개:
  (a) 불일치 51문제에 추가 stochastic 생성(step 2)을 하면 어디까지 갈 수 있나?
      → 기존 샘플 기준 oracle이 상한의 근사치다.
  (b) 합의 413문제 중 틀린 64문제는 버려지는 것인가?
      → 여기 oracle이 크면 합의 게이트 자체가 손해다.
"""
import json
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/holdout/official_holdout_464_clean.csv"

DET = {
    "grpo96": "outputs/grpo_3145_passrate94_steps96_holdout464_retry2048.jsonl",
    "verbose": "outputs/verbose_distill_holdout464_retry2048.jsonl",
    "h3145": "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl",
}
POOLS = {
    "grpo96": "outputs/self_consistency_grpo96_n8_holdout464_seed20260826.jsonl",
    "verbose": "outputs/self_consistency_verbose_n8_holdout464_seed20260827.jsonl",
    "h3145": "outputs/self_consistency_hybrid3145_n8_holdout500.jsonl",
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


def main():
    with open(GOLD, encoding="utf-8-sig", newline="") as f:
        gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(f)}
    ids = sorted(gold)

    det = {}
    for k, p in DET.items():
        d = {}
        for line in open(ROOT / p):
            r = json.loads(line)
            d[r["id"]] = norm(r.get("prediction"))
        det[k] = d

    pool = {}
    for k, p in POOLS.items():
        d = {}
        for line in open(ROOT / p):
            r = json.loads(line)
            d[r["id"]] = [s for s in (norm(x) for x in r.get("sample_predictions", [])) if s is not None]
        pool[k] = d

    g, v = det["grpo96"], det["verbose"]
    agree = [i for i in ids if g.get(i) is not None and g[i] == v.get(i)]
    disagree = [i for i in ids if i not in set(agree)]

    def pool_all(i, keys):
        c = Counter()
        for k in keys:
            c.update(pool[k].get(i, []))
        return c

    print(f"합의 {len(agree)} / 불일치 {len(disagree)}\n")

    # (a) 불일치 51문제 헤드룸
    print("=== (a) 불일치 51문제: 어디까지 회수 가능한가 ===")
    cur = sum(1 for i in disagree if g[i] == gold[i])
    det_or = sum(1 for i in disagree if gold[i] in (g.get(i), v.get(i)))
    gv_or = sum(1 for i in disagree if gold[i] in set(pool["grpo96"].get(i, [])) | set(pool["verbose"].get(i, [])))
    all_or = sum(1 for i in disagree if gold[i] in set(pool_all(i, pool)) | {g.get(i), v.get(i), det["h3145"].get(i)})
    print(f"  현재(grpo96 채택)          {cur:3d}/51")
    print(f"  det 2개 중 완벽 선택 상한   {det_or:3d}/51")
    print(f"  기존 16샘플 oracle          {gv_or:3d}/51")
    print(f"  기존 24샘플+det oracle      {all_or:3d}/51")
    print(f"  → step 2가 완벽해도 전체 상한은 349+{all_or} = {349+all_or}/464 근처")
    print()

    # (b) 합의 413문제에서 버려지는 몫
    print("=== (b) 합의 413문제: 게이트가 버리는 몫 ===")
    aw = [i for i in agree if g[i] != gold[i]]
    a_pool_or = sum(1 for i in aw if gold[i] in set(pool_all(i, pool)))
    print(f"  합의했는데 틀린 문제        {len(aw):3d}개")
    print(f"  그 중 기존 24샘플 oracle에 정답이 있는 문제  {a_pool_or:3d}개  ← 합의 게이트를 씌우면 이 구간은 손도 못 댄다")
    print()

    # (c) 전면 적용 규칙 재확인 + 합의 구간 threshold 차등
    print("=== (c) 전면 pool 규칙 vs 합의 게이트 ===")

    def plur(c):
        if not c:
            return None
        r = c.most_common()
        if len(r) > 1 and r[0][1] == r[1][1]:
            return None
        return r[0][0]

    def score(pred):
        return sum(1 for i in ids if pred.get(i) == gold[i])

    keys3 = ["grpo96", "verbose", "h3145"]
    for mc in (10, 11, 12):
        pred = {}
        for i in ids:
            p = plur(pool_all(i, keys3))
            pred[i] = p if (p is not None and pool_all(i, keys3)[p] >= mc) else g.get(i)
        print(f"  전면 3계보24 min{mc:<2d} (동률/미달→grpo96)      {score(pred)}/464")

    keys2 = ["grpo96", "verbose"]
    for mc in (7, 8):
        pred = {}
        for i in ids:
            p = plur(pool_all(i, keys2))
            pred[i] = p if (p is not None and pool_all(i, keys2)[p] >= mc) else g.get(i)
        print(f"  전면 2계보16 min{mc:<2d} (동률/미달→grpo96)      {score(pred)}/464")

    # 차등: 합의 구간은 높은 threshold, 불일치 구간은 낮은 threshold
    print()
    print("  [차등] 합의 구간 threshold a / 불일치 구간 threshold d — 2계보 16샘플")
    best = []
    for a in (0, 8, 10, 12, 14, 16, 99):
        row = []
        for d in (0, 4, 5, 6, 7, 8):
            pred = {}
            aset = set(agree)
            for i in ids:
                c = pool_all(i, keys2)
                p = plur(c)
                th = a if i in aset else d
                if a == 99 and i in aset:
                    pred[i] = g.get(i)
                elif p is not None and c[p] >= th:
                    pred[i] = p
                else:
                    pred[i] = g.get(i)
            s = score(pred)
            row.append(s)
            best.append((s, a, d))
        print(f"    a={a:<3d} " + " ".join(f"d={d}:{s}" for d, s in zip((0, 4, 5, 6, 7, 8), row)))
    top = max(best)
    print(f"    → 최고 {top[0]}/464 (합의 threshold={top[1]}, 불일치 threshold={top[2]})")


if __name__ == "__main__":
    main()
