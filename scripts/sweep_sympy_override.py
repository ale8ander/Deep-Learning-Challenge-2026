"""SymPy TIR 증거 기반 오버라이드 스윕 — 기준선 665 등가(387/464).

규칙 가족:
  A) sympy verified 유일최빈 >= k & 현행과 다름 & 구코드가드(구 verified 가 현행 우세면 취소)
  B) 구+신 verified 합산 유일최빈 >= k & 현행과 다름
  C) A + ck150 N=64 최빈 일치 보조 조건
발사 바: 델타 >= +8 & 변경 >= 20 & calib/valid 양수 (미달 시 표만 제시, 발사는 사용자 결정)
"""
import argparse
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


def load_ver(rels):
    d = {}
    for rel in rels:
        for r in jl(rel):
            c = d.setdefault(r["id"], Counter())
            for a, v in (r.get("verified_counts") or {}).items():
                a = tnorm(a)
                if a is not None:
                    c[a] += v
    return d


ap = argparse.ArgumentParser()
ap.add_argument("--sympy", default="outputs/sympy_tir8_holdout464.jsonl")
args = ap.parse_args()

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
    c = Counter(norm(x) for x in r["predictions"] if norm(x) is not None)
    tp = c.most_common()
    ck64[r["id"]] = str(tp[0][0]) if tp and (len(tp) == 1 or tp[0][1] > tp[1][1]) else None

old_ver = load_ver(["outputs/tir_sc8_holdout464_vote3_to60.jsonl",
                    "outputs/tir_repair1_holdout464_vote3.jsonl",
                    "outputs/tir_repair_nocode_holdout464_vote3.jsonl",
                    "outputs/tirc_hybrid3145_holdout464_vote45.jsonl"])
sym_ver = load_ver([args.sympy])


def guard_old(i, cur, new):
    c = old_ver.get(i, Counter())
    return not (c and c.get(tnorm(cur), 0) > c.get(tnorm(new), 0))


f665 = dict(f660)
for i in ids:
    m, cnt = ck8.get(i, (None, 0))
    if m is not None and cnt >= 5 and m != str(f660[i]) and support[i] <= 4 and guard_old(i, f660[i], m):
        f665[i] = m
b665 = sum(1 for i in ids if str(f665[i]) == str(gold[i]))
print(f"기준선 665 등가: {b665}/464 | sympy 커버 문제 {len(sym_ver)}개, "
      f"검증표 총합 {sum(sum(c.values()) for c in sym_ver.values())}\n")


def judge(label, new):
    s = sum(1 for i in ids if str(new[i]) == str(gold[i]))
    ch = [i for i in ids if new[i] != f665[i]]
    g = sum(1 for i in ch if str(new[i]) == str(gold[i]))
    rg = sum(1 for i in ch if str(f665[i]) == str(gold[i]))
    cn = sum(1 for i in ids if half(i) and str(new[i]) == str(gold[i]))
    cb = sum(1 for i in ids if half(i) and str(f665[i]) == str(gold[i]))
    print(f"{label:<40} {s-b665:+d}  변경 {len(ch)} g{g}/r{rg}  "
          f"calib {cn-cb:+d} valid {(s-cn)-(b665-cb):+d}")
    return s - b665, len(ch), cn - cb, (s - cn) - (b665 - cb)


results = []
for k in (2, 3, 4, 5, 6):
    # A) sympy 단독 + 구코드가드
    a = dict(f665)
    for i in ids:
        c = sym_ver.get(i, Counter())
        tp = c.most_common()
        if tp and tp[0][1] >= k and (len(tp) == 1 or tp[0][1] > tp[1][1]):
            m = str(tp[0][0])
            if m != str(a[i]) and guard_old(i, a[i], m):
                a[i] = m
    results.append((f"A k>={k} sympy+구가드", judge(f"A k>={k} (sympy 유일최빈, 구코드가드)", a), a))
    # B) 합산
    b = dict(f665)
    for i in ids:
        c = old_ver.get(i, Counter()) + sym_ver.get(i, Counter())
        tp = c.most_common()
        if tp and tp[0][1] >= k and (len(tp) == 1 or tp[0][1] > tp[1][1]):
            m = str(tp[0][0])
            if m != str(b[i]):
                b[i] = m
    results.append((f"B k>={k} 합산", judge(f"B k>={k} (구+sympy 합산 유일최빈)", b), b))
    # C) A + ck64 일치
    cvar = dict(f665)
    for i in ids:
        c = sym_ver.get(i, Counter())
        tp = c.most_common()
        if tp and tp[0][1] >= k and (len(tp) == 1 or tp[0][1] > tp[1][1]):
            m = str(tp[0][0])
            if m != str(cvar[i]) and guard_old(i, cvar[i], m) and ck64.get(i) == m:
                cvar[i] = m
    results.append((f"C k>={k} +ck64일치", judge(f"C k>={k} (A + ck64 최빈 일치)", cvar), cvar))

print("\n발사 바: 델타>=+8 & 변경>=20 & calib/valid 양수")
ok = [(n, r) for n, r, _ in results if r[0] >= 8 and r[1] >= 20 and r[2] > 0 and r[3] > 0]
print(f"통과: {len(ok)}개")
for n, r in ok:
    print("  ", n, r)
