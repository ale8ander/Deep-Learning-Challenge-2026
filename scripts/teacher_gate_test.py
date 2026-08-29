"""새 우물 게이트 — 임의의 N=8 샘플 파일을 삼중 게이트 신호로 끼워 665 등가 대비 판정.

이긴 템플릿(18절) 그대로: support<=4 x [신호 N=8 유일최빈 >= T표] x 코드가드.
ck150 대신 새 독립 계보(teacher32b 등)를 신호로 쓴다. ck 게이트가 이미 먹은 문제는
665 등가에 반영돼 있으므로, 여기서 잡히는 건 전부 순증분이다.

사용: /usr/bin/python3 scripts/teacher_gate_test.py --n8 outputs/teacher_n8_holdout464_seedX.jsonl
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402
from tir_common import normalize as tnorm  # noqa: E402

VOTERS = [
    "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl",
    "outputs/hybrid_3244_holdout464_retry2048.jsonl",
    "outputs/external_3000_holdout464_retry2048.jsonl",
    "outputs/hybrid_4145_holdout464_retry2048.jsonl",
    "outputs/hybrid_3145_verify_holdout464_retry2048.jsonl",
]
SC_MAIN = "outputs/self_consistency_confidence_n8_holdout464.jsonl"
CK_N8 = "outputs/ck150_n8_holdout464_seed20260920.jsonl"
FINAL660 = "outputs/champion660_holdout464_equivalent.jsonl"
TIR_POOLS = [
    "outputs/tir_sc8_holdout464_vote3_to60.jsonl",
    "outputs/tir_repair1_holdout464_vote3.jsonl",
    "outputs/tir_repair_nocode_holdout464_vote3.jsonl",
    "outputs/tirc_hybrid3145_holdout464_vote45.jsonl",
]


def jl(rel):
    return [json.loads(l) for l in open(ROOT / rel) if l.strip()]


ap = argparse.ArgumentParser()
ap.add_argument("--n8", required=True, help="predictions 리스트를 가진 N=8 jsonl")
ap.add_argument("--label", default="teacher")
args = ap.parse_args()

gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(
    open(ROOT / "data/holdout/official_holdout_464_clean.csv", encoding="utf-8-sig"))}
ids = [i for i in gold if gold[i] is not None]

# --- 배포 체인 (base623 -> 660 -> 665 등가) ---
vp = []
for rel in VOTERS:
    d = {}
    for r in jl(rel):
        p = r.get("prediction")
        d[r["id"]] = norm(p) if p is not None else norm(extract_v2(r.get("response")))
    vp.append(d)
hyb = {}
for r in jl(SC_MAIN):
    resp = r.get("responses") or []
    hyb[r["id"]] = ([norm(extract_v2(t)) for t in resp] if resp
                    else [norm(x) for x in (r.get("sample_predictions") or [])])
support = {}
for i in ids:
    votes = [d.get(i) for d in vp]
    c = Counter(v for v in votes if v is not None)
    support[i] = c.most_common(1)[0][1] if c else 0
final660 = {r["id"]: norm(r.get("prediction")) for r in jl(FINAL660)}
ck = {r["id"]: [norm(x) for x in r["predictions"]] for r in jl(CK_N8)}
ver = {}
for rel in TIR_POOLS:
    for r in jl(rel):
        c = ver.setdefault(r["id"], Counter())
        for a, v in (r.get("verified_counts") or {}).items():
            a = tnorm(a)
            if a is not None:
                c[a] += v


def umode(samples, min_v):
    tp = Counter(x for x in samples if x is not None).most_common()
    if tp and not (len(tp) > 1 and tp[0][1] == tp[1][1]) and tp[0][1] >= min_v:
        return tp[0][0], tp[0][1]
    return None, 0


def guard_ok(i, cur, mode):
    vc = ver.get(i, Counter())
    return not (vc and vc.get(tnorm(str(cur)), 0) > vc.get(tnorm(str(mode)), 0))


base665 = dict(final660)
for i in ids:
    if support[i] <= 4:
        m, nv = umode(ck.get(i, []), 5)
        if m is not None and str(m) != str(base665[i]) and guard_ok(i, base665[i], m):
            base665[i] = m
B = sum(1 for i in ids if base665[i] == gold[i])
print(f"sanity: 665 등가 = {B}/464 (기대 387)")

sig = {r["id"]: [norm(x) for x in r["predictions"]] for r in jl(args.n8)}
cov = sum(1 for i in ids if i in sig)
print(f"{args.label} N=8 커버리지: {cov}/{len(ids)}\n")


def run(tag, cond):
    flips = []
    for i in ids:
        if support[i] > 4:
            continue
        ok, m = cond(i)
        if ok and str(m) != str(base665[i]) and guard_ok(i, base665[i], m):
            flips.append((i, m))
    pred = dict(base665)
    for i, a in flips:
        pred[i] = a
    s = sum(1 for i in ids if pred[i] == gold[i])
    g = sum(1 for i, a in flips if a == gold[i])
    rg = sum(1 for i, a in flips if base665[i] == gold[i])
    print(f"[{tag}] 교체 {len(flips):>2}개 -> {s}/464 ({s-B:+d}, gain {g}/reg {rg})")
    for i, a in flips:
        mark = "O" if a == gold[i] else ("X<-O" if base665[i] == gold[i] else "X<-X")
        print(f"   {i}: {base665[i]} -> {a}  {mark} (sup{support[i]})")
    return flips


# 본선: teacher 단독 게이트 (문턱 스윕)
for t in (5, 6, 7, 8):
    run(f"{args.label}>={t} & 코드가드", lambda i, t=t: (lambda m, nv: (m is not None, m))(*umode(sig.get(i, []), t)))
    print()

# 교집합 강화: teacher>=5 이면서 ck150 도 같은 답에 >=4표 (이중 새 우물)
def both(i):
    m, _ = umode(sig.get(i, []), 5)
    if m is None:
        return False, None
    cm, cv = umode(ck.get(i, []), 4)
    return (cm is not None and str(cm) == str(m)), m
run(f"{args.label}>=5 & ck>=4 동일답 & 코드가드", both)
