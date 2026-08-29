"""표수4~5 밴드(홀드아웃 69문제)에서 TIR 규칙을 스윕한다 — 마지막 미개척 제출 레버.

현행 배포 규칙은 이 구간에서 **risky>=1 인 문제만** TIR 을 발동한다(831 기준 109문제 중 26개).
나머지 83문제는 손대지 않았고, 831 환산 오답이 이 구간에 34개 있다.

질문 두 가지를 동시에 판정한다:
  (a) risky 게이트를 **없애고 전면 적용**하면 좋아지는가?
  (b) min-count 는 몇이 최적인가?

⚠️ 69문제는 87문제보다도 작다. 표준오차가 ±4 를 넘을 수 있으므로
   **델타만 보지 말고 gain/regression 과 calib/valid 분할을 함께** 본다.
   포개진 규칙끼리는 짝비교(discordant pairs)가 주변 SE 보다 훨씬 예민하다.
"""
import csv, hashlib, itertools, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/DLC")
sys.path.insert(0, str(ROOT / "scripts"))
from tir_common import normalize as n  # noqa: E402

POOLS = {
    "A100": "outputs/tirc_hybrid3145_holdout464_vote45.jsonl",  # 옛 A100 산출물(60문제만)
    "V45a": "outputs/tir_v45_a100_holdout_vote45.jsonl",
    "V45r": "outputs/tir_v45_r1_holdout_vote45.jsonl",
    "V45n": "outputs/tir_v45_nc_holdout_vote45.jsonl",
}


def half(i):
    return int(hashlib.sha256(("split:" + i).encode()).hexdigest()[:8], 16) % 2 == 0


def main():
    band = list(csv.DictReader(
        open(ROOT / "data/holdout/holdout464_vote45_band.csv", encoding="utf-8-sig")))
    gold = {r["id"]: n(r["answer"]) for r in band}
    meta = json.load(open(ROOT / "outputs/holdout464_vote45_meta.json"))
    champ = {k: n(v["champ"]) for k, v in meta.items()}
    risky = {k: v["nrisky"] for k, v in meta.items()}
    ids = [r["id"] for r in band if gold[r["id"]] is not None and champ.get(r["id"]) is not None]

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
    nr = sum(1 for i in ids if risky.get(i, 0) >= 1)
    print(f"표수4~5 밴드 {len(ids)}문제 / 챔피언 {base} / risky>=1 {nr}문제")
    print(f"사용 가능 풀: {', '.join(avail)}\n")
    if not avail:
        print("풀이 하나도 없다 — 생성 먼저"); return

    def apply(cb, mc, risky_only):
        out = {}
        for i in ids:
            if risky_only and risky.get(i, 0) < 1:
                out[i] = champ[i]
                continue
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
            for risky_only in (True, False):
                for mc in range(2, 3 * len(cb) + 1):
                    pr = apply(cb, mc, risky_only)
                    s = sum(1 for i in ids if pr[i] == gold[i])
                    g = sum(1 for i in ids if pr[i] != champ[i] and pr[i] == gold[i])
                    rg = sum(1 for i in ids if pr[i] != champ[i] and champ[i] == gold[i])
                    f = sum(1 for i in ids if pr[i] != champ[i])
                    cn = sum(1 for i in ids if half(i) and pr[i] == gold[i])
                    cb2 = sum(1 for i in ids if half(i) and champ[i] == gold[i])
                    rows.append((s - base, "+".join(cb), 8 * len(cb), mc,
                                 "risky" if risky_only else "전면",
                                 g, rg, f, cn - cb2, (s - cn) - (base - cb2), pr))

    rows.sort(key=lambda x: (-x[0], -min(x[8], x[9])))
    print(f"{'조합':<18}{'N':>3}{'mc':>4}{'게이트':>7}{'델타':>6}"
          f"{'gain':>6}{'reg':>5}{'발동':>5}{'calib':>7}{'valid':>7}")
    for r in rows[:16]:
        print(f"{r[1]:<18}{r[2]:>3}{r[3]:>4}{r[4]:>7}{r[0]:>+6}"
              f"{r[5]:>6}{r[6]:>5}{r[7]:>5}{r[8]:>+7}{r[9]:>+7}")

    # 현행(risky 전용, 최선 mc)과 최고 후보의 짝비교
    cur = max((r for r in rows if r[4] == "risky"), key=lambda x: x[0], default=None)
    best = rows[0]
    if cur and (best[1], best[3], best[4]) != (cur[1], cur[3], cur[4]):
        disc = [i for i in ids if best[10][i] != cur[10][i]]
        bw = sum(1 for i in disc if best[10][i] == gold[i])
        cw = sum(1 for i in disc if cur[10][i] == gold[i])
        print(f"\n짝비교 — 최고({best[1]} mc{best[3]} {best[4]}) "
              f"vs 현행형({cur[1]} mc{cur[3]} risky)")
        print(f"  답 갈린 {len(disc)}문제 | 최고 {bw} / 현행형 {cw} / 둘다틀림 {len(disc)-bw-cw}")
        print("  승부가 1:1 수준이면 표본 부족이지 '동률'이 아니다.")

    print("\n⚠️ 69문제는 87문제보다 작다. 델타 단독으로 판단하지 말고 gain/reg 와")
    print("   calib/valid 가 **둘 다 양수**인지, 짝비교가 명확한지를 함께 볼 것.")
    print("   채택 시: build_merged16_submission.py --vote45-min-count <mc>")
    print("   (risky 게이트 제거가 이기면 스크립트의 nrisky 조건도 함께 풀어야 한다)")


if __name__ == "__main__":
    main()
