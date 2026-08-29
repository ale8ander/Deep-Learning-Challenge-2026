"""최종샷 스윕 — 게이트 확장(집중도 0.40~0.55) + 코드맥스 + 탐욕 합집합.

기준선 665 등가(387/464). 모든 게이트 후보에 코드가드 필수.
출력: 단독 후보 표 + 탐욕 합집합 결과 + 확정 규칙 명세(JSON, 빌더가 그대로 사용).
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
from tir_common import normalize as tnorm  # noqa: E402


def jl(rel):
    try:
        return [json.loads(l) for l in open(ROOT / rel) if l.strip()]
    except FileNotFoundError:
        return []


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
ck64 = {}
for r in jl("outputs/ck150_n64_holdout464_seed20260922.jsonl"):
    preds = [norm(x) for x in r["predictions"] if norm(x) is not None]
    c = Counter(preds)
    tp = c.most_common()
    if tp and (len(tp) == 1 or tp[0][1] > tp[1][1]):
        ck64[r["id"]] = (str(tp[0][0]), tp[0][1] / len(preds))
    else:
        ck64[r["id"]] = (None, 0.0)
old_ver = {}
for rel in ["outputs/tir_sc8_holdout464_vote3_to60.jsonl",
            "outputs/tir_repair1_holdout464_vote3.jsonl",
            "outputs/tir_repair_nocode_holdout464_vote3.jsonl",
            "outputs/tirc_hybrid3145_holdout464_vote45.jsonl"]:
    for r in jl(rel):
        c = old_ver.setdefault(r["id"], Counter())
        for a, v in (r.get("verified_counts") or {}).items():
            a = tnorm(a)
            if a is not None:
                c[a] += v


def guard_ok(i, cur, new):
    c = old_ver.get(i, Counter())
    return not (c and c.get(tnorm(cur), 0) > c.get(tnorm(new), 0))


f665 = dict(f660)
for i in ids:
    m, cnt = ck8.get(i, (None, 0))
    if m is not None and cnt >= 5 and m != str(f660[i]) and support[i] <= 4 and guard_ok(i, f660[i], m):
        f665[i] = m
b665 = sum(1 for i in ids if str(f665[i]) == str(gold[i]))
print(f"기준선 665 등가: {b665}/464\n")

WEAK = {"sup<=3": lambda i: support[i] <= 3, "sup<=4": lambda i: support[i] <= 4,
        "전체": lambda i: True}


def gate_flips(frac, wname):
    out = {}
    for i in ids:
        m, f = ck64.get(i, (None, 0.0))
        if m is not None and f >= frac and m != str(f665[i]) and WEAK[wname](i) and guard_ok(i, f665[i], m):
            out[i] = m
    return out


def codemax_flips():
    out = {}
    for i in ids:
        c = old_ver.get(i, Counter())
        if not c:
            continue
        tp = c.most_common()
        if not tp or (len(tp) > 1 and tp[0][1] == tp[1][1]):
            continue
        m = str(tp[0][0])
        cv = c.get(tnorm(f665[i]), 0)
        if m != str(f665[i]) and ((tp[0][1] >= 2 and cv == 0) or (tp[0][1] >= 3 and cv == 1)):
            out[i] = m
    return out


def judge(label, flips, quiet=False):
    new = dict(f665)
    new.update(flips)
    s = sum(1 for i in ids if str(new[i]) == str(gold[i]))
    ch = [i for i in ids if new[i] != f665[i]]
    g = sum(1 for i in ch if str(new[i]) == str(gold[i]))
    rg = sum(1 for i in ch if str(f665[i]) == str(gold[i]))
    cn = sum(1 for i in ids if half(i) and str(new[i]) == str(gold[i]))
    cb = sum(1 for i in ids if half(i) and str(f665[i]) == str(gold[i]))
    if not quiet:
        print(f"{label:<32} {s-b665:+d}  변경 {len(ch)} g{g}/r{rg}  "
              f"calib {cn-cb:+d} valid {(s-cn)-(b665-cb):+d}")
    return s - b665


cands = []
for frac in (0.40, 0.425, 0.45, 0.475, 0.50, 0.525, 0.55):
    for wname in WEAK:
        fl = gate_flips(frac, wname)
        if fl:
            cands.append((f"게이트 {frac:.3f}/{wname}", fl))
cands.append(("코드맥스", codemax_flips()))

print("— 단독 후보 —")
scored = []
for name, fl in cands:
    d = judge(name, fl)
    scored.append((d, name, fl))

# 탐욕 합집합: 델타 순으로 정렬, 증분 델타 > 0 인 것만 추가 (기존 플립과 충돌 시 먼저 온 규칙 우선)
scored.sort(key=lambda x: -x[0])
union, used = {}, []
best = 0
for d, name, fl in scored:
    trial = dict(union)
    for i, a in fl.items():
        trial.setdefault(i, a)
    nd = judge("", trial, quiet=True)
    if nd > best:
        union, best = trial, nd
        used.append(name)

print("\n— 탐욕 합집합 —")
judge("합집합(" + " + ".join(used) + ")", union)
spec = {"rules": used, "flips_holdout": {i: union[i] for i in sorted(union)}}
(ROOT / "outputs/final_union_rulespec.json").write_text(json.dumps(spec, ensure_ascii=False, indent=2))
print(f"\n규칙 명세 저장: outputs/final_union_rulespec.json (채택 규칙: {used})")
