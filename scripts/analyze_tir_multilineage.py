"""TIR 3계보 합산 — 코드검증된 답을 한 풀에 모아 다수결한다.

배경: 오늘 반복 확인된 패턴이 "서로 다르게 틀리는 계보를 섞으면 크게 번다"이고,
오전에 순수 CoT 3계보 24샘플이 holdout464에서 371(역대 최고)을 냈다.
여기서는 같은 레시피에 **코드 실행 검증**이라는 훨씬 강한 필터가 붙는다.

각 계보는 같은 Qwen2.5-3B 베이스에 우리가 학습시킨 LoRA 어댑터만 다르다.
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

    pools = {}
    for name, path in LINEAGES.items():
        p = ROOT / path
        if not p.exists():
            print(f"[없음] {name}")
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
    print(f"대상 {len(ids)}문제  baseline {b_all}  계보 {sorted(pools)}")
    print()

    print("=== 계보별 코드검증 오라클 ===")
    for name, d in pools.items():
        orc = sum(1 for i in ids if gold[i] in d[i])
        print(f"  {name:12s} {orc:3d}/{len(ids)}")
    print()

    print("=== 조합별 오라클 (합집합) ===")
    names = sorted(pools)
    for size in range(2, len(names) + 1):
        for combo in combinations(names, size):
            orc = sum(1 for i in ids if any(gold[i] in pools[n][i] for n in combo))
            print(f"  {'+'.join(combo):40s} {orc:3d}/{len(ids)}")
    print()

    print("=== 합산 다수결 (min-count 스윕) ===")
    print(f"{'조합':30s} {'mc':>3s} {'점수':>8s} {'차이':>5s} {'gain':>5s} {'reg':>4s} {'채택':>5s} {'정밀':>6s} {'cal':>5s} {'val':>5s}")
    best = None
    for size in range(1, len(names) + 1):
        for combo in combinations(names, size):
            for mc in range(1, 4 * size + 1):
                pred, adopted, hit = {}, 0, 0
                for i in ids:
                    c = Counter()
                    for n in combo:
                        c.update(pools[n][i])
                    top = c.most_common()
                    if top and top[0][1] >= mc and not (len(top) > 1 and top[0][1] == top[1][1]):
                        pred[i] = top[0][0]; adopted += 1; hit += int(pred[i] == gold[i])
                    else:
                        pred[i] = base[i]
                n_ok = sum(1 for i in ids if pred[i] == gold[i])
                g = sum(1 for i in ids if base[i] != gold[i] and pred[i] == gold[i])
                rg = sum(1 for i in ids if base[i] == gold[i] and pred[i] != gold[i])
                nc = sum(1 for i in calib if pred[i] == gold[i]) - sum(1 for i in calib if base[i] == gold[i])
                nv = sum(1 for i in valid if pred[i] == gold[i]) - sum(1 for i in valid if base[i] == gold[i])
                row = (n_ok, combo, mc, g, rg, adopted, hit / adopted if adopted else 0, nc, nv)
                if best is None or n_ok > best[0]:
                    best = row
                if n_ok - b_all >= 10:  # 의미 있는 것만 출력
                    print(f"{'+'.join(combo):30s} {mc:3d} {n_ok:4d}/{len(ids)} {n_ok-b_all:+5d} {g:5d} {rg:4d} "
                          f"{adopted:5d} {row[6]:5.0%} {nc:+5d} {nv:+5d}")
    print()
    if best:
        n_ok, combo, mc, g, rg, adopted, prec, nc, nv = best
        print(f"최고: {'+'.join(combo)} min-count {mc} -> {n_ok}/{len(ids)} ({n_ok-b_all:+d}), "
              f"gain {g} reg {rg}, 채택 {adopted} 정밀도 {prec:.0%}, calib {nc:+d} valid {nv:+d}")
        print()
        print("참고 — 단일 계보 hybrid3145 min-count 2: +13 (gain 14/reg 1, 정밀도 65%)  ← 현재 제출 규칙")


if __name__ == "__main__":
    main()
