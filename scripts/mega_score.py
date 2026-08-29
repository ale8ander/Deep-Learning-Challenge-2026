"""메가 홀드아웃 2,000 채점 — 최종 제출 후보 파이프라인 전 비교.

후보 체인 (배포 규칙 그대로 재구성):
  base   : hybrid3145 단독 greedy (참고)
  c623   : 5-voter(유일최대>=2, else hybrid) + support4 SC override(hybrid N=8 mode>=4)
  c660   : c623 + 표수<=3 24샘플(mc2) + 표수4~5 risky 16샘플(mc2)
  c665   : c660 + 삼중 게이트 (ck8 mode>=5 & sup<=4 & 코드가드)
  c664v3 : c665 + 게이트 v3 (ck64 frac>=0.425 & sup<=4 & 코드가드)
"""
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/DLC")
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402
from tir_common import normalize as tnorm  # noqa: E402

O = ROOT / "outputs/mega"


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(
    open(ROOT / "data/holdout/mega_holdout_2000.csv", encoding="utf-8-sig"))}
ids = [i for i in gold if gold[i] is not None]

voters = {}
for name in ["hybrid3145", "h3244", "ext3000", "h4145", "verify"]:
    d = {}
    for r in jl(O / f"voter_{name}.jsonl"):
        p = r.get("prediction")
        d[r["id"]] = norm(p) if p is not None else norm(extract_v2(r.get("response")))
    voters[name] = d
bands = json.load(open(O / "mega_bands.json"))
support, votes = bands["support"], bands["votes"]

sc_samples = {}
for r in jl(O / "sc_hybrid_n8.jsonl"):
    sc_samples[r["id"]] = [norm(extract_v2(t)) for t in (r.get("responses") or [])]


def vercounts(paths):
    d = {}
    for p in paths:
        for r in jl(O / p):
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
    c = Counter(norm(x) for x in r["predictions"] if norm(x) is not None)
    tp = c.most_common()
    ck8[r["id"]] = (str(tp[0][0]), tp[0][1]) if tp and (len(tp) == 1 or tp[0][1] > tp[1][1]) else (None, 0)
ck64 = {}
for r in jl(O / "ck150_n64_sup4.jsonl"):
    preds = [norm(x) for x in r["predictions"] if norm(x) is not None]
    c = Counter(preds)
    tp = c.most_common()
    if preds and tp and (len(tp) == 1 or tp[0][1] > tp[1][1]):
        ck64[r["id"]] = (str(tp[0][0]), tp[0][1] / len(preds))
    else:
        ck64[r["id"]] = (None, 0.0)

# c623
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

# c660
c660 = dict(c623)
for i in ids:
    pool = v3pool if votes[i] <= 3 else (v45pool if (votes[i] in (4, 5) and bands["nrisky"][i] >= 1) else None)
    if pool is None:
        continue
    c = pool.get(i, Counter())
    tp = c.most_common()
    if tp and tp[0][1] >= 2 and (len(tp) == 1 or tp[0][1] > tp[1][1]):
        c660[i] = str(tp[0][0])


def guard(i, cur, new):
    c = allver.get(i, Counter())
    return not (c and c.get(tnorm(cur), 0) > c.get(tnorm(new), 0))


# c665
c665 = dict(c660)
for i in ids:
    m, cnt = ck8.get(i, (None, 0))
    if m is not None and cnt >= 5 and support[i] <= 4 and m != str(c660[i]) and guard(i, c660[i], m):
        c665[i] = m
# c664v3
c664 = dict(c665)
for i in ids:
    m, f = ck64.get(i, (None, 0.0))
    if m is not None and f >= 0.425 and support[i] <= 4 and m != str(c665[i]) and guard(i, c665[i], m):
        c664[i] = m

cands = [("base(hybrid greedy)", voters["hybrid3145"]), ("c623 챔피언", c623),
         ("c660 pool24", c660), ("c665 삼중게이트", c665), ("c664 게이트v3", c664)]
print(f"메가 홀드아웃 {len(ids)}문제 (신선, GRPO 학습분 제외)\n")
prev = None
for name, pred in cands:
    s = sum(1 for i in ids if str(pred.get(i)) == str(gold[i]))
    line = f"{name:<22} {s}/{len(ids)}  ({s/len(ids)*100:.2f}%)"
    if prev is not None:
        pn, pp = prev
        ch = [i for i in ids if str(pred.get(i)) != str(pp.get(i))]
        g = sum(1 for i in ch if str(pred.get(i)) == str(gold[i]))
        rg = sum(1 for i in ch if str(pp.get(i)) == str(gold[i]))
        line += f"   [vs {pn}: 변경 {len(ch)}, gain {g}/reg {rg}, 순 {g-rg:+d}]"
    print(line)
    prev = (name.split()[0], pred)
