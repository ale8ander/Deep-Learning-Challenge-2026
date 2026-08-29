"""게이트 v4 후보 — 공짜(CPU) 확인 2종, 665 등가(387/464) 기준.

A. support5 만장일치 게이트: sup==5 에서 ck150 N=8 이 8/8(또는 >=7)로 현행 답에
   반대하고 코드가드가 반대하지 않으면 교체. (어제 게이트는 sup<=4 만 건드림 — 미개척)
B. 이중 확인 게이트: sup<=4, ck150 4표(단독으론 Public -1로 죽은 층)이지만
   독립 계보(verbose N=8 / hybrid N=8)의 최빈값이 ck 답과 일치할 때만 + 코드가드.

⚠️ 변경 수가 작으면(C-3) 홀드아웃 델타는 못 믿는다 — 개수·순도만 본다.
"""
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
CK_N64 = "outputs/ck150_n64_holdout464_seed20260922.jsonl"
VERBOSE_N8 = "outputs/self_consistency_verbose_n8_holdout464_seed20260827.jsonl"
FINAL660 = "outputs/champion660_holdout464_equivalent.jsonl"
TIR_POOLS = [
    "outputs/tir_sc8_holdout464_vote3_to60.jsonl",
    "outputs/tir_repair1_holdout464_vote3.jsonl",
    "outputs/tir_repair_nocode_holdout464_vote3.jsonl",
    "outputs/tirc_hybrid3145_holdout464_vote45.jsonl",
]


def jl(rel):
    return [json.loads(l) for l in open(ROOT / rel) if l.strip()]


def sc_samples(rel, key_resp="responses", key_pred="sample_predictions"):
    d = {}
    for r in jl(rel):
        resp = r.get(key_resp) or []
        d[r["id"]] = ([norm(extract_v2(t)) for t in resp] if resp
                      else [norm(x) for x in (r.get(key_pred) or r.get("predictions") or [])])
    return d


gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(
    open(ROOT / "data/holdout/official_holdout_464_clean.csv", encoding="utf-8-sig"))}
ids = [i for i in gold if gold[i] is not None]

# --- 배포 체인 재구성 (base623 -> 660 -> 665 등가) ---
vp = []
for rel in VOTERS:
    d = {}
    for r in jl(rel):
        p = r.get("prediction")
        d[r["id"]] = norm(p) if p is not None else norm(extract_v2(r.get("response")))
    vp.append(d)
hyb = sc_samples(SC_MAIN)
support, base623 = {}, {}
for i in ids:
    votes = [d.get(i) for d in vp]
    c = Counter(v for v in votes if v is not None)
    support[i] = c.most_common(1)[0][1] if c else 0
    winners = [a for a, n_ in c.items() if n_ == max(c.values())] if c else []
    base623[i] = winners[0] if (len(winners) == 1 and max(c.values()) >= 2) else votes[0]
    if support[i] == 4:
        tp = Counter(x for x in hyb.get(i, []) if x is not None).most_common()
        if tp and tp[0][1] >= 4 and (len(tp) == 1 or tp[0][1] > tp[1][1]):
            base623[i] = tp[0][0]
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


def ck_mode(i, min_votes):
    tp = Counter(x for x in ck.get(i, []) if x is not None).most_common()
    if tp and not (len(tp) > 1 and tp[0][1] == tp[1][1]) and tp[0][1] >= min_votes:
        return tp[0][0], tp[0][1]
    return None, 0


def codeguard_ok(i, cur, mode):
    vc = ver.get(i, Counter())
    return not (vc and vc.get(tnorm(str(cur)), 0) > vc.get(tnorm(str(mode)), 0))


# 665 등가 = 660 + 삼중 게이트(sup<=4, ck>=5, 코드가드)
base665 = dict(final660)
for i in ids:
    if support[i] <= 4:
        m, nv = ck_mode(i, 5)
        if m is not None and str(m) != str(base665[i]) and codeguard_ok(i, base665[i], m):
            base665[i] = m
B = sum(1 for i in ids if base665[i] == gold[i])
print(f"sanity: 665 등가 = {B}/464 (기대 387)\n")

verbose = sc_samples(VERBOSE_N8)
n64 = {r["id"]: [norm(x) for x in (r.get("predictions") or [])] for r in jl(CK_N64)} \
    if (ROOT / CK_N64).exists() else {}


def report(tag, flips):
    """flips: list of (id, new_answer)"""
    pred = dict(base665)
    for i, a in flips:
        pred[i] = a
    s = sum(1 for i in ids if pred[i] == gold[i])
    g = sum(1 for i, a in flips if a == gold[i])
    rg = sum(1 for i, a in flips if base665[i] == gold[i])
    det = [(i, a, "O" if a == gold[i] else ("X<-O" if base665[i] == gold[i] else "X<-X"))
           for i, a in flips]
    print(f"[{tag}] 교체 {len(flips)}개 -> {s}/464 ({s-B:+d}, gain {g}/reg {rg})")
    for i, a, mark in det:
        print(f"   {i}: {base665[i]} -> {a}  {mark}")
    print()


# --- A. support5 만장일치 게이트 ---
for min_v in (8, 7):
    flips = []
    for i in ids:
        if support[i] == 5:
            m, nv = ck_mode(i, min_v)
            if m is not None and str(m) != str(base665[i]) and codeguard_ok(i, base665[i], m):
                flips.append((i, m))
    report(f"A: sup5 & ck>={min_v} & 코드가드", flips)

# --- B. ck 4표 + 독립 계보 확인 ---
for conf_name, conf in (("verbose", verbose), ("hybridSC", hyb)):
    for conf_min in (4, 5):
        flips = []
        for i in ids:
            if support[i] <= 4:
                m, nv = ck_mode(i, 4)
                if m is None or nv != 4 or str(m) == str(base665[i]):
                    continue
                cs = Counter(x for x in conf.get(i, []) if x is not None)
                tp = cs.most_common()
                if not (tp and str(tp[0][0]) == str(m) and tp[0][1] >= conf_min
                        and (len(tp) == 1 or tp[0][1] > tp[1][1])):
                    continue
                if codeguard_ok(i, base665[i], m):
                    flips.append((i, m))
        report(f"B: sup<=4 & ck==4표 & {conf_name} 최빈>={conf_min} 일치 & 코드가드", flips)

# --- 참고: A 를 N=64 집중도로 재확인 (재료 있으면) ---
if n64:
    flips = []
    for i in ids:
        if support[i] == 5:
            m, nv = ck_mode(i, 7)
            if m is None or str(m) == str(base665[i]):
                continue
            c64 = Counter(x for x in n64.get(i, []) if x is not None)
            conc = c64.get(m, 0) / max(1, sum(c64.values()))
            if conc >= 0.6 and codeguard_ok(i, base665[i], m):
                flips.append((i, m))
    report("A+: sup5 & ck8>=7 & N64 집중도>=0.6 & 코드가드", flips)
