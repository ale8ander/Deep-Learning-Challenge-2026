"""선택 격차의 구조 분석 — 놓치는 문제들이 '아깝게' 지는가, '압도적으로' 지는가.

배경: 4계보 32샘플에서 코드검증 오라클은 48/87인데 실제 채택은 30/87이다.
18문제에서 정답이 후보로 존재하는데 못 고른다. 이 18개를 회수할 수 있는지는
**표 차이의 크기**에 달렸다.

  - 정답 3표 vs 오답 4표  -> 약한 직교 신호로도 뒤집힌다. selector 투자 가치 있음
  - 정답 1표 vs 오답 7표  -> 어떤 재랭킹도 못 뒤집는다. 투자 무의미

부수적으로 계보 breadth(몇 개 계보가 그 답을 지지하는가)가 신호가 되는지도 함께 본다.
CoT에서는 기각됐지만(CONTEXT 7절) TIR 후보에서는 미검증이다.
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tir_inference import normalize  # noqa: E402

SUBSET = ROOT / "data/holdout/holdout464_vote3.csv"
LINEAGES = {
    "hybrid3145": "outputs/tir_sc8_holdout464_vote3_to60.jsonl",
    "grpo96": "outputs/tir_sc8_holdout464_grpo96.jsonl",
    "verbose": "outputs/tir_sc8_holdout464_verbose.jsonl",
    "tirsft": "outputs/tir_sc8_holdout464_tirsft_fixed.jsonl",
}


def main():
    gold = {r["id"]: normalize(r["answer"])
            for r in csv.DictReader(open(SUBSET, encoding="utf-8-sig"))}

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

    ids = [i for i in sorted(gold) if all(i in d for d in pools.values())]
    names = sorted(pools)
    print(f"대상 {len(ids)}문제, 계보 {names} (문제당 최대 {8*len(names)}샘플)\n")

    have, picked, missed = [], [], []
    for i in ids:
        total = Counter()
        breadth = Counter()  # 답 -> 그 답을 검증한 계보 수
        for n in names:
            total.update(pools[n][i])
            for ans in pools[n][i]:
                breadth[ans] += 1
        if gold[i] not in total:
            continue
        have.append(i)
        top = total.most_common()
        winner = top[0][0] if not (len(top) > 1 and top[0][1] == top[1][1]) else None
        if winner == gold[i]:
            picked.append(i)
        else:
            missed.append((i, total, breadth, winner))

    print(f"정답이 후보로 존재: {len(have)}/{len(ids)}")
    print(f"  다수결이 맞힘: {len(picked)}")
    print(f"  놓침         : {len(missed)}   <- 회수 대상\n")

    print("=== 놓친 문제들의 표 구조 ===")
    print(f"{'id':16s} {'정답표':>5s} {'1위표':>5s} {'차이':>5s} {'정답순위':>7s} {'정답계보':>7s} {'1위계보':>6s}")
    diffs, ranks = [], []
    for i, total, breadth, winner in missed:
        gv = total[gold[i]]
        tv = total.most_common(1)[0][1]
        ranked = [a for a, _ in total.most_common()]
        rank = ranked.index(gold[i]) + 1
        diffs.append(tv - gv)
        ranks.append(rank)
        wb = breadth[winner] if winner is not None else 0
        print(f"{i:16s} {gv:5d} {tv:5d} {tv-gv:5d} {rank:7d} {breadth[gold[i]]:7d} {wb:6d}")

    print()
    print("=== 회수 난이도 요약 ===")
    print(f"  표 차이 분포: {dict(sorted(Counter(diffs).items()))}")
    print(f"  정답 순위 분포: {dict(sorted(Counter(ranks).items()))}")
    close = sum(1 for d in diffs if d <= 2)
    rank2 = sum(1 for r in ranks if r == 2)
    print(f"  표 차이 2 이하(아깝게 짐): {close}/{len(missed)}")
    print(f"  정답이 2위: {rank2}/{len(missed)}")
    print()

    # breadth 신호 검사: 정답과 1위 오답의 계보 수 비교
    win_b, gold_b = [], []
    for i, total, breadth, winner in missed:
        if winner is None:
            continue
        win_b.append(breadth[winner])
        gold_b.append(breadth[gold[i]])
    if win_b:
        print("=== breadth 신호 (몇 개 계보가 지지하는가) ===")
        print(f"  놓친 문제에서 1위 오답의 평균 계보 수: {sum(win_b)/len(win_b):.2f}")
        print(f"  같은 문제에서 정답의 평균 계보 수    : {sum(gold_b)/len(gold_b):.2f}")
        better = sum(1 for a, b in zip(gold_b, win_b) if a > b)
        print(f"  정답이 더 많은 계보의 지지를 받은 경우: {better}/{len(win_b)}"
              f"  -> {'breadth로 일부 뒤집을 수 있다' if better >= 4 else 'breadth는 신호가 아니다'}")
    print()

    # 맞힌 문제와 비교: 표 차이가 신뢰도 지표가 되는가
    ok_margin = []
    for i in picked:
        total = Counter()
        for n in names:
            total.update(pools[n][i])
        top = total.most_common()
        second = top[1][1] if len(top) > 1 else 0
        ok_margin.append(top[0][1] - second)
    if ok_margin:
        print("=== 참고: 맞힌 문제의 1위-2위 표 차이 ===")
        print(f"  평균 {sum(ok_margin)/len(ok_margin):.2f}, 분포 {dict(sorted(Counter(ok_margin).items()))}")


if __name__ == "__main__":
    main()
