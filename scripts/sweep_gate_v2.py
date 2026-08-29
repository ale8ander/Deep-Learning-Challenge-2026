"""삼중 게이트 v2 스윕 — N=64 심층 샘플로 확신 문턱을 정밀 탐색.

기준선: 665 등가(387/464). 조건 그리드: 집중도(유일최빈 비율) × support 조건,
코드가드는 전 조합 고정. 채택 바: 델타>=+5 & 변경>=20 & calib/valid 양수.
"""
import csv, hashlib, json, sys
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/DLC")
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402
from tir_common import normalize as tnorm  # noqa: E402


def jl(rel):
    return [json.loads(l) for l in open(ROOT / rel) if l.strip()]


def half(i):
    return int(hashlib.sha256(("split:" + i).encode()).hexdigest()[:8], 16) % 2 == 0


gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(
    open(ROOT / "data/holdout/official_holdout_464_clean.csv", encoding="utf-8-sig"))}
ids = [i for i in gold if gold[i] is not None]
f660 = {r["id"]: norm(r.get("prediction")) for r in jl("outputs/champion660_holdout464_equivalent.jsonl")}

vp = []
for rel in ["outputs/holdout464_hybrid_3145_baseline_predictions.jsonl",
            "outputs/hybrid_3244_holdout464_retry2048.jsonl",
            "outputs/external_3000_holdout464_retry2048.jsonl",
            "outputs/hybrid_4145_holdout464_retry2048.jsonl",
            "outputs/hybrid_3145_verify_holdout464_retry2048.jsonl"]:
    d = {}
    for r in jl(rel):
        p = r.get("prediction")
        d[r["id"]] = norm(p) if p is not None else norm(extract_v2(r.get("response")))
    vp.append(d)
support = {}
for i in ids:
    c = Counter(v for v in (d.get(i) for d in vp) if v is not None)
    support[i] = c.most_common(1)[0][1] if c else 0

ck8 = {}
for r in jl("outputs/ck150_n8_holdout464_seed20260920.jsonl"):
    c = Counter(norm(x) for x in r["predictions"] if norm(x) is not None)
    tp = c.most_common()
    ck8[r["id"]] = (str(tp[0][0]), tp[0][1]) if tp and (len(tp) == 1 or tp[0][1] > tp[1][1]) else (None, 0)

ver = {}
for rel in ["outputs/tir_sc8_holdout464_vote3_to60.jsonl",
            "outputs/tir_repair1_holdout464_vote3.jsonl",
            "outputs/tir_repair_nocode_holdout464_vote3.jsonl",
            "outputs/tirc_hybrid3145_holdout464_vote45.jsonl"]:
    for r in jl(rel):
        c = ver.setdefault(r["id"], Counter())
        for a, v in (r.get("verified_counts") or {}).items():
            a = tnorm(a)
            if a is not None:
                c[a] += v


def guard_ok(i, cur, new):
    c = ver.get(i, Counter())
    return not (c and c.get(tnorm(cur), 0) > c.get(tnorm(new), 0))


f665 = dict(f660)
for i in ids:
    m, cnt = ck8.get(i, (None, 0))
    if m is not None and cnt >= 5 and m != str(f660[i]) and support[i] <= 4 and guard_ok(i, f660[i], m):
        f665[i] = m
b665 = sum(1 for i in ids if str(f665[i]) == str(gold[i]))
print(f"기준선 665 등가: {b665}/464\n")

ck64 = {}
for r in jl("outputs/ck150_n64_holdout464_seed20260922.jsonl"):
    preds = [norm(x) for x in r["predictions"] if norm(x) is not None]
    c = Counter(preds)
    tp = c.most_common()
    if tp and (len(tp) == 1 or tp[0][1] > tp[1][1]):
        ck64[r["id"]] = (str(tp[0][0]), tp[0][1], len(preds))
    else:
        ck64[r["id"]] = (None, 0, len(preds))

WEAK = {"sup<=3": lambda i: support[i] <= 3, "sup<=4": lambda i: support[i] <= 4,
        "sup==4": lambda i: support[i] == 4, "전체": lambda i: True}
rows = []
for frac in (0.50, 0.55, 0.60, 0.65, 0.70, 0.75, 0.80):
    for wname, wf in WEAK.items():
        new = dict(f665)
        ch = []
        for i in ids:
            m, cnt, tot = ck64.get(i, (None, 0, 0))
            if (m is not None and tot >= 48 and cnt / tot >= frac and m != str(new[i])
                    and wf(i) and guard_ok(i, new[i], m)):
                new[i] = m
                ch.append(i)
        s = sum(1 for i in ids if str(new[i]) == str(gold[i]))
        g = sum(1 for i in ch if str(new[i]) == str(gold[i]))
        rg = sum(1 for i in ch if str(f665[i]) == str(gold[i]))
        cn = sum(1 for i in ids if half(i) and str(new[i]) == str(gold[i]))
        cb = sum(1 for i in ids if half(i) and str(f665[i]) == str(gold[i]))
        rows.append((s - b665, frac, wname, len(ch), g, rg, cn - cb, (s - cn) - (b665 - cb)))
rows.sort(key=lambda x: (-x[0], -x[3]))
print(f"{'집중도':>5} {'조건':<8}{'델타':>5}{'변경':>5}{'gain':>5}{'reg':>4}{'calib':>6}{'valid':>6}")
for r in rows[:16]:
    print(f"{r[1]:>5.2f} {r[2]:<8}{r[0]:>+5}{r[3]:>5}{r[4]:>5}{r[5]:>4}{r[6]:>+6}{r[7]:>+6}")
ok = [r for r in rows if r[0] >= 5 and r[3] >= 20 and r[6] > 0 and r[7] > 0]
print(f"\n채택 바(델타>=+5 & 변경>=20 & calib/valid 양수) 통과: {len(ok)}개")
for r in ok[:5]:
    print("  ", r)
