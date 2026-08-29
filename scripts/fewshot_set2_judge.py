"""few-shot 세트2 판정 — 672 등가 기준, 세트1과 동일한 사전 등록 규칙.

규칙(고정): 포인터 greedy 답 != 현행답, 16샘플 표>=4, 상대우위, 자기재현 fs8>=2.
기준선: 665등가 + 세트1 규칙 12플립 = 672 등가 (~396/464 예상).
겹침 분석: 세트2가 세트1과 같은 문제를 다시 여는지(중복) vs 새 문제를 여는지.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/DLC")
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402

VOTERS = [
    "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl",
    "outputs/hybrid_3244_holdout464_retry2048.jsonl",
    "outputs/external_3000_holdout464_retry2048.jsonl",
    "outputs/hybrid_4145_holdout464_retry2048.jsonl",
    "outputs/hybrid_3145_verify_holdout464_retry2048.jsonl",
]
FINAL660 = "outputs/champion660_holdout464_equivalent.jsonl"
CK_N8_OLD = "outputs/ck150_n8_holdout464_seed20260920.jsonl"
LP_POOLS = ["outputs/ck150_n8lp_holdout464_seed20260924.jsonl",
            "outputs/h3145_n8lp_holdout464_seed20260924.jsonl"]
TIR_POOLS = [
    "outputs/tir_sc8_holdout464_vote3_to60.jsonl",
    "outputs/tir_repair1_holdout464_vote3.jsonl",
    "outputs/tir_repair_nocode_holdout464_vote3.jsonl",
    "outputs/tirc_hybrid3145_holdout464_vote45.jsonl",
]
SET1_G = "outputs/fewshot3_h3145_holdout464.jsonl"
SET1_8 = "outputs/fewshot3_h3145_n8_holdout464_seed20260925.jsonl"
SET2_G = "outputs/fewshot_set2_h3145_holdout464.jsonl"
SET2_8 = "outputs/fewshot_set2_h3145_n8_holdout464_seed20260926.jsonl"


def jl(rel):
    return [json.loads(l) for l in open(ROOT / rel) if l.strip()]


from tir_common import normalize as tnorm  # noqa: E402

gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(
    open(ROOT / "data/holdout/official_holdout_464_clean.csv", encoding="utf-8-sig"))}
ids = [i for i in gold if gold[i] is not None]
calib = {i for i in ids if int(hashlib.sha256(i.encode()).hexdigest(), 16) % 2 == 0}

vp = []
for rel in VOTERS:
    d = {}
    for r in jl(rel):
        p = r.get("prediction")
        d[r["id"]] = norm(p) if p is not None else norm(extract_v2(r.get("response")))
    vp.append(d)
support = {}
for i in ids:
    c = Counter(v for v in (d.get(i) for d in vp) if v is not None)
    support[i] = c.most_common(1)[0][1] if c else 0
final660 = {r["id"]: norm(r.get("prediction")) for r in jl(FINAL660)}
ck_old = {r["id"]: [norm(x) for x in r["predictions"]] for r in jl(CK_N8_OLD)}
ver = {}
for rel in TIR_POOLS:
    for r in jl(rel):
        c = ver.setdefault(r["id"], Counter())
        for a, v in (r.get("verified_counts") or {}).items():
            a = tnorm(a)
            if a is not None:
                c[a] += v


def umode(preds, min_v=1):
    tp = Counter(x for x in preds if x is not None).most_common()
    if tp and not (len(tp) > 1 and tp[0][1] == tp[1][1]) and tp[0][1] >= min_v:
        return tp[0][0], tp[0][1]
    return None, 0


def guard_ok(i, cur, mode):
    vc = ver.get(i, Counter())
    return not (vc and vc.get(tnorm(str(cur)), 0) > vc.get(tnorm(str(mode)), 0))


base665 = dict(final660)
for i in ids:
    if support[i] <= 4:
        m, nv = umode(ck_old.get(i, []), 5)
        if m is not None and str(m) != str(base665[i]) and guard_ok(i, base665[i], m):
            base665[i] = m

pool16 = {}
for rel in LP_POOLS:
    for r in jl(rel):
        pool16.setdefault(r["id"], []).extend(
            None if p is None else str(norm(p)) for p in r["predictions"])


def votes(i, a):
    return sum(1 for p in pool16.get(i, []) if p == a)


def load_g(rel):
    return {r["id"]: (None if r["prediction"] is None else str(r["prediction"]))
            for r in jl(rel)}


def load_8(rel):
    return {r["id"]: r.get("predictions", []) for r in jl(rel)}


def rule_flips(base, g, n8):
    flips = []
    for i in ids:
        a = g.get(i)
        if a is None or str(a) == str(base[i]):
            continue
        if votes(i, a) < 4:
            continue
        if votes(i, a) <= votes(i, str(base[i])):
            continue
        if sum(1 for p in n8.get(i, []) if p == a) < 2:
            continue
        flips.append((i, a))
    return flips


# 672 등가 = 665등가 + 세트1 플립
s1 = rule_flips(base665, load_g(SET1_G), load_8(SET1_8))
base672 = dict(base665)
for i, a in s1:
    base672[i] = a
B = sum(1 for i in ids if base672[i] == gold[i])
print(f"672 등가 = {B}/464 (세트1 플립 {len(s1)}개 반영)")

g2, n82 = load_g(SET2_G), load_8(SET2_8)
s2 = rule_flips(base672, g2, n82)
g_ = sum(1 for i, a in s2 if a == str(gold[i]))
r_ = sum(1 for i, a in s2 if base672[i] == gold[i])
dc = sum(1 for i, a in s2 if i in calib and a == str(gold[i])) - \
     sum(1 for i, a in s2 if i in calib and base672[i] == gold[i])
print(f"\n세트2 규칙 적용: 교체 {len(s2)} gain {g_}/reg {r_} 델타 {g_-r_:+d} "
      f"(cal {dc:+d}/val {(g_-r_)-dc:+d})")
s1ids = {i for i, _ in s1}
for i, a in s2:
    m = "O" if a == str(gold[i]) else ("X<-O" if base672[i] == gold[i] else "X<-X")
    print(f"  {i}: {base672[i]} -> {a} {m} (sup{support[i]}, 표16 {votes(i,a)}vs{votes(i,str(base672[i]))}, "
          f"fs8 {sum(1 for p in n82.get(i,[]) if p==a)}{', 세트1도 플립' if i in s1ids else ''}")
# 참고: 세트2 단독 성능과 신규 오라클
tot2 = sum(1 for i in ids if g2.get(i) == str(gold[i]))
new_open = sum(1 for i in ids if g2.get(i) == str(gold[i]) and base672[i] != gold[i])
print(f"\n세트2 greedy 단독 {tot2}/464 | 672등가 오답 중 세트2 정답 {new_open}개")
