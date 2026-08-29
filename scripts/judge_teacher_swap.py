"""teacher32b base 교체 C-1 재판정 — 게이트 B(챔피언 대비 +6)의 착시 여부 확인.

게이트 B 는 챔피언(18/87) 기준이었지만, 현행 656 배포 규칙은 이 밴드에서
A100+NC1 16샘플 mc3 로 챔피언 +15 수준이다. base 교체가 성립하려면 teacher
단독 풀이 hybrid 단독 풀(+11)을 넘어야 하고, 최종적으로 현행 배포(+15)를
넘을 전망이 있어야 한다. 규칙은 sweep_pools_holdout.py 와 동일
(unique-mode >= mc, 미발동 시 챔피언 fallback).
"""
import csv, hashlib, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/DLC")
sys.path.insert(0, str(ROOT / "scripts"))
from tir_common import normalize as n  # noqa: E402

POOLS = {
    "A100": "outputs/tir_sc8_holdout464_vote3_to60.jsonl",
    "R1":   "outputs/tir_repair1_holdout464_vote3.jsonl",
    "NC1":  "outputs/tir_repair_nocode_holdout464_vote3.jsonl",
    "T":    "outputs/teachersft_tir8_holdout87.jsonl",
}


def half(i):
    return int(hashlib.sha256(("split:" + i).encode()).hexdigest()[:8], 16) % 2 == 0


def load_pool(rel):
    d = {}
    for l in open(ROOT / rel):
        r = json.loads(l)
        c = Counter()
        for a, v in (r.get("verified_counts") or {}).items():
            a = n(a)
            if a is not None:
                c[a] += v
        d[r["id"]] = c
    return d


def main():
    gold = {r["id"]: n(r["answer"]) for r in csv.DictReader(
        open(ROOT / "data/holdout/holdout464_vote3.csv", encoding="utf-8-sig"))}
    champ = {}
    for l in open(ROOT / "outputs/champion_holdout464_equivalent.jsonl"):
        r = json.loads(l)
        champ[r["id"]] = n(r.get("prediction"))
    ids = [i for i in gold if gold[i] is not None and i in champ]
    pools = {k: load_pool(rel) for k, rel in POOLS.items()}
    base = sum(1 for i in ids if champ[i] == gold[i])
    print(f"홀드아웃 표수<=3 {len(ids)}문제 / 챔피언 {base}\n")

    def apply(cb, mc):
        out = {}
        for i in ids:
            t = Counter()
            for k in cb:
                t += pools[k].get(i, Counter())
            tp = t.most_common()
            pick = tp[0][0] if tp and tp[0][1] >= mc and (
                len(tp) == 1 or tp[0][1] > tp[1][1]) else None
            out[i] = pick if pick is not None else champ[i]
        return out

    def report(label, cb, mc):
        pr = apply(cb, mc)
        s = sum(1 for i in ids if pr[i] == gold[i])
        g = sum(1 for i in ids if pr[i] != champ[i] and pr[i] == gold[i])
        rg = sum(1 for i in ids if pr[i] != champ[i] and champ[i] == gold[i])
        f = sum(1 for i in ids if pr[i] != champ[i])
        cn = sum(1 for i in ids if half(i) and pr[i] == gold[i])
        cb_ = sum(1 for i in ids if half(i) and champ[i] == gold[i])
        orc = sum(1 for i in ids
                  if any(gold[i] in pools[k].get(i, {}) for k in cb))
        print(f"{label:<24} 델타 {s-base:+d}  gain {g} reg {rg} 발동 {f}  "
              f"calib {cn-cb_:+d} valid {(s-cn)-(base-cb_):+d}  오라클 {orc}")
        return pr

    print("— 단독 풀 (동일 규칙 mc2) —")
    a100 = report("A100 단독 (hybrid)", ("A100",), 2)
    report("NC1 단독 (hybrid)", ("NC1",), 2)
    report("R1 단독 (hybrid)", ("R1",), 2)
    t = report("T 단독 (teacher32b)", ("T",), 2)

    print("\n— 배포/대기 규칙 —")
    report("A100+NC1 mc3 (현행656)", ("A100", "NC1"), 3)
    report("A100+R1+NC1 mc2 (P0)", ("A100", "R1", "NC1"), 2)

    disc = [i for i in ids if t[i] != a100[i]]
    tw = sum(1 for i in disc if t[i] == gold[i])
    aw = sum(1 for i in disc if a100[i] == gold[i])
    print(f"\n짝비교 T mc2 vs A100 mc2 — 갈린 {len(disc)}개 | T 맞음 {tw} / "
          f"A100 맞음 {aw} / 둘다틀림 {len(disc)-tw-aw}")
    print("\n판정: base 교체가 성립하려면 T 단독이 hybrid 단독(+11)을 짝비교로")
    print("명확히 이기고, 교체 후 조합이 현행(+15) 이상일 전망이어야 한다.")


if __name__ == "__main__":
    main()
