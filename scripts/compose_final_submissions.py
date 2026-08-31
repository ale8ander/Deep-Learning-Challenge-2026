"""최종 테스트/메가 공용 — 배포 체인(623/660/665/672) 재구성 + 제출 CSV 조립.

mega_score.py 의 체인 로직에 672 층(few-shot 포인터 게이트)을 추가하고,
입력·산출 디렉터리를 인자로 받는다. 입력 CSV 에 answer 열이 있으면(=메가)
후보별 점수도 채점한다. 없으면(=최종 테스트) CSV 조립만 한다.

  # 메가 채점 (672 포함)
  python3 scripts/compose_final_submissions.py --input data/holdout/mega_holdout_2000.csv \
      --materials outputs/mega
  # 최종 테스트 제출본
  python3 scripts/compose_final_submissions.py --input data/final_test.csv \
      --materials outputs/final --emit c672=submissions/final_c672.csv \
      --emit c623=submissions/final_c623.csv
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/DLC")
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402
from tir_common import normalize as tnorm  # noqa: E402


def S(x):
    return None if x is None else str(x)


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--materials", required=True)
    ap.add_argument("--emit", action="append", default=[],
                    help="candidate=path.csv (반복 가능)")
    args = ap.parse_args()
    O = Path(args.materials)

    rows = list(csv.DictReader(open(args.input, encoding="utf-8-sig")))
    ids = [r["id"] for r in rows]
    gold = {r["id"]: S(norm(r["answer"])) for r in rows if r.get("answer")}

    voters = {}
    for name in ["hybrid3145", "h3244", "ext3000", "h4145", "verify"]:
        d = {}
        for r in jl(O / f"voter_{name}.jsonl"):
            p = r.get("prediction")
            d[r["id"]] = S(norm(p) if p is not None else norm(extract_v2(r.get("response"))))
        voters[name] = d
    bands = json.load(open(O / "mega_bands.json"))
    support, votes_sc, nrisky = bands["support"], bands["votes"], bands["nrisky"]

    sc_samples = {r["id"]: [S(norm(extract_v2(t))) for t in (r.get("responses") or [])]
                  for r in jl(O / "sc_hybrid_n8.jsonl")}

    def vercounts(paths):
        d = {}
        for p in paths:
            fp = O / p
            if not fp.exists():
                continue
            for r in jl(fp):
                c = d.setdefault(r["id"], Counter())
                for a, v in (r.get("verified_counts") or {}).items():
                    a = tnorm(a)
                    if a is not None:
                        c[a] += v
        return d

    v3pool = vercounts(["tir_a100_vote3.jsonl", "tir_r1_vote3.jsonl", "tir_nc_vote3.jsonl"])
    v45pool = vercounts(["tir_a100_v45r.jsonl", "tir_nc_v45r.jsonl"])
    allver = vercounts(["tir_a100_vote3.jsonl", "tir_r1_vote3.jsonl", "tir_nc_vote3.jsonl",
                        "tir_a100_v45r.jsonl", "tir_nc_v45r.jsonl"])

    ck8 = {}
    for r in jl(O / "ck150_n8_sup4.jsonl"):
        c = Counter(S(norm(x)) for x in r["predictions"] if norm(x) is not None)
        tp = c.most_common()
        ck8[r["id"]] = (tp[0][0], tp[0][1]) if tp and (len(tp) == 1 or tp[0][1] > tp[1][1]) else (None, 0)

    # ── c623 ──
    c623 = {}
    for i in ids:
        vs = [voters[n].get(i) for n in ["hybrid3145", "h3244", "ext3000", "h4145", "verify"]]
        c = Counter(v for v in vs if v is not None)
        w = [a for a, n_ in c.items() if n_ == max(c.values())] if c else []
        c623[i] = w[0] if (len(w) == 1 and max(c.values()) >= 2) else vs[0]
        if support[i] == 4:
            cc = Counter(x for x in sc_samples.get(i, []) if x is not None)
            tp = cc.most_common()
            if tp and tp[0][1] >= 4 and (len(tp) == 1 or tp[0][1] > tp[1][1]):
                c623[i] = tp[0][0]

    # ── c660 ──
    c660 = dict(c623)
    for i in ids:
        pool = v3pool if votes_sc[i] <= 3 else (
            v45pool if (votes_sc[i] in (4, 5) and nrisky[i] >= 1) else None)
        if pool is None:
            continue
        tp = pool.get(i, Counter()).most_common()
        if tp and tp[0][1] >= 2 and (len(tp) == 1 or tp[0][1] > tp[1][1]):
            c660[i] = S(tp[0][0])

    def guard(i, cur, new):
        c = allver.get(i, Counter())
        return not (c and c.get(tnorm(str(cur)), 0) > c.get(tnorm(str(new)), 0))

    # ── c665 (삼중 게이트) ──
    c665 = dict(c660)
    for i in ids:
        m, cnt = ck8.get(i, (None, 0))
        if m is not None and cnt >= 5 and support[i] <= 4 and m != S(c660[i]) \
                and guard(i, c660[i], m):
            c665[i] = m

    # ── c672 (few-shot 포인터 게이트) ──
    fs_g = {r["id"]: S(norm(r.get("prediction"))) for r in jl(O / "fs3_greedy.jsonl")}
    fs_8 = {r["id"]: [S(norm(p)) for p in r.get("predictions", [])]
            for r in jl(O / "fs3_n8.jsonl")}
    pool16 = {}
    for f in ["ck150_n8lp.jsonl", "h3145_n8lp.jsonl"]:
        for r in jl(O / f):
            pool16.setdefault(r["id"], []).extend(
                S(norm(p)) if p is not None else None for p in r["predictions"])

    def p16(i, a):
        return sum(1 for p in pool16.get(i, []) if p == a)

    c672 = dict(c665)
    fired = 0
    for i in ids:
        a = fs_g.get(i)
        if a is None or a == S(c672[i]):
            continue
        if p16(i, a) < 4 or p16(i, a) <= p16(i, S(c672[i])):
            continue
        if sum(1 for p in fs_8.get(i, []) if p == a) < 2:
            continue
        c672[i] = a
        fired += 1
    print(f"fs 게이트 발동 {fired} (831에서는 21이었음 — 2,000 환산 기대 ~50)")

    cands = {"c623": c623, "c660": c660, "c665": c665, "c672": c672}

    if gold:
        print(f"\n채점 ({len(gold)}문제):")
        prev = None
        order = [("base", voters["hybrid3145"])] + list(cands.items())
        for name, pred in order:
            s = sum(1 for i in ids if i in gold and S(pred.get(i)) == gold[i])
            line = f"  {name:<6} {s}/{len(gold)} ({s / len(gold) * 100:.2f}%)"
            if prev:
                pn, pp = prev
                ch = [i for i in ids if S(pred.get(i)) != S(pp.get(i))]
                g = sum(1 for i in ch if i in gold and S(pred.get(i)) == gold[i])
                rg = sum(1 for i in ch if i in gold and S(pp.get(i)) == gold[i])
                line += f"  [vs {pn}: 변경 {len(ch)}, g{g}/r{rg}, 순 {g - rg:+d}]"
            print(line)
            prev = (name, pred)

    for spec in args.emit:
        name, out = spec.split("=", 1)
        pred = cands[name]
        missing = [i for i in ids if pred.get(i) is None]
        with open(out, "w", newline="", encoding="utf-8") as f:
            w = csv.writer(f)
            w.writerow(["id", "answer"])
            for i in ids:
                v = pred.get(i)
                w.writerow([i, int(str(v)) if v is not None else 0])
        print(f"emit {name} -> {out} ({len(ids)}행, 결측 {len(missing)})")
        if missing:
            print(f"  ⚠️ 결측 id 예시: {missing[:5]}")


if __name__ == "__main__":
    main()
