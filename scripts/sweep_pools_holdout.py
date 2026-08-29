"""홀드아웃87(표수<=3)에서 TIR 풀 조합 x min-count 를 전수 스윕한다.

왜 필요한가: 831 에는 정답이 없어 규칙을 판정할 수 없다. 풀을 더 쌓을지,
어떤 조합으로 합칠지는 **반드시 여기서 먼저** 정해야 한다 (CONTEXT C-4).

판정 기준은 **챔피언 대비**다 (C-1). hybrid_3145 대비로 재고 환산하지 않는다.
⚠️ 87문제 표준오차 ±3.5~3.9 (C-3). 델타 차이 4 미만은 실효과로 읽지 말 것.
   그래서 짝지어 비교(discordant pairs)도 함께 찍는다 — 포개진 조합끼리는
   주변 표준오차보다 짝비교가 훨씬 예민하다.
"""
import csv, hashlib, itertools, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/DLC")
sys.path.insert(0, str(ROOT / "scripts"))
from tir_common import normalize as n  # noqa: E402

POOLS = {
    "A100": "outputs/tir_sc8_holdout464_vote3_to60.jsonl",
    "R1":   "outputs/tir_repair1_holdout464_vote3.jsonl",
    "NC1":  "outputs/tir_repair_nocode_holdout464_vote3.jsonl",
    "NC2":  "outputs/verifier_holdout87_hybrid3145.jsonl",
    "R2":   "outputs/tir_r2_holdout464_vote3.jsonl",
    "NC3":  "outputs/tir_nc3_holdout464_vote3.jsonl",
}


def half(i):
    return int(hashlib.sha256(("split:" + i).encode()).hexdigest()[:8], 16) % 2 == 0


def main():
    gold = {r["id"]: n(r["answer"]) for r in csv.DictReader(
        open(ROOT / "data/holdout/holdout464_vote3.csv", encoding="utf-8-sig"))}
    champ = {}
    for l in open(ROOT / "outputs/champion_holdout464_equivalent.jsonl"):
        r = json.loads(l)
        champ[r["id"]] = n(r.get("prediction"))
    ids = [i for i in gold if gold[i] is not None and i in champ]

    pools, avail = {}, []
    for k, rel in POOLS.items():
        p = ROOT / rel
        if not p.exists():
            continue
        d = {}
        for l in open(p):
            r = json.loads(l)
            c = Counter()
            for a, v in (r.get("verified_counts") or {}).items():
                a = n(a)
                if a is not None:
                    c[a] += v
            d[r["id"]] = c
        pools[k] = d
        avail.append(k)

    base = sum(1 for i in ids if champ[i] == gold[i])
    print(f"홀드아웃 표수<=3 {len(ids)}문제 / 챔피언 {base}")
    print(f"사용 가능 풀: {', '.join(avail)}\n")

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

    rows = []
    for r_ in range(1, len(avail) + 1):
        for cb in itertools.combinations(avail, r_):
            if "A100" not in cb:      # A100 은 항상 포함 (현행 제출본의 기반)
                continue
            orc = sum(1 for i in ids
                      if any(gold[i] in pools[k].get(i, {}) for k in cb))
            for mc in range(2, 3 * len(cb) + 1):
                pr = apply(cb, mc)
                s = sum(1 for i in ids if pr[i] == gold[i])
                g = sum(1 for i in ids if pr[i] != champ[i] and pr[i] == gold[i])
                rg = sum(1 for i in ids if pr[i] != champ[i] and champ[i] == gold[i])
                f = sum(1 for i in ids if pr[i] != champ[i])
                cn = sum(1 for i in ids if half(i) and pr[i] == gold[i])
                cb_ = sum(1 for i in ids if half(i) and champ[i] == gold[i])
                rows.append((s - base, "+".join(cb), 8 * len(cb), mc, g, rg, f,
                             cn - cb_, (s - cn) - (base - cb_), orc, pr))

    rows.sort(key=lambda x: (-x[0], -x[7] - x[8]))
    print(f"{'조합':<26}{'N':>3}{'mc':>4}{'델타':>6}{'gain':>6}{'reg':>5}"
          f"{'발동':>5}{'calib':>7}{'valid':>7}{'오라클':>7}")
    for r in rows[:16]:
        print(f"{r[1]:<26}{r[2]:>3}{r[3]:>4}{r[0]:>+6}{r[4]:>6}{r[5]:>5}"
              f"{r[6]:>5}{r[7]:>+7}{r[8]:>+7}{r[9]:>7}")

    # 현행 배포(A100+NC1 mc2)와 최고 후보의 짝비교 — 포개진 비교는 주변 SE 보다 예민하다
    cur = next((r for r in rows if r[1] == "A100+NC1" and r[3] == 2), None)
    best = rows[0]
    if cur and best[1] != cur[1]:
        disc = [i for i in ids if best[10][i] != cur[10][i]]
        bw = sum(1 for i in disc if best[10][i] == gold[i])
        cw = sum(1 for i in disc if cur[10][i] == gold[i])
        print(f"\n짝비교 — 최고({best[1]} mc{best[3]}) vs 현행(A100+NC1 mc2)")
        print(f"  답 갈린 문제 {len(disc)}개 | 최고가 맞음 {bw} / 현행이 맞음 {cw} "
              f"/ 둘다틀림 {len(disc)-bw-cw}")
        print("  (승부가 1:1 수준이면 표본이 부족한 것이지 '동률'이 아니다)")
    print("\n⚠️ 87문제 표준오차 ±3.5~3.9. 델타 4 미만 차이는 실효과로 읽지 말 것.")


if __name__ == "__main__":
    main()
