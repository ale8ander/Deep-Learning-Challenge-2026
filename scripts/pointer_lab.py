"""포인터 실험실 — 672 등가(홀드아웃) 기준으로 게이트 구조의 남은 변형을 전수 판정.

R0 무포인터: pool16 최빈답이 챔피언답보다 표가 많으면(마진 스윕) 교체
R1 다중 포인터 합의: 보유 greedy 포인터들 중 >=P 개가 같은 비챔피언 답 & pool16 확인
R2 확신 가중 마진: raw 표 대신 exp(4*min_group) 가중 우위로 재판정

모든 답은 str 로 정규화 (세트2 판정기의 타입 버그 수정판).
"""
import csv
import hashlib
import json
import math
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path("/workspace/DLC")
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402
from tir_common import normalize as tnorm  # noqa: E402


def S(x):
    return None if x is None else str(x)


def jl(rel):
    return [json.loads(l) for l in open(ROOT / rel) if l.strip()]


VOTERS = [
    "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl",
    "outputs/hybrid_3244_holdout464_retry2048.jsonl",
    "outputs/external_3000_holdout464_retry2048.jsonl",
    "outputs/hybrid_4145_holdout464_retry2048.jsonl",
    "outputs/hybrid_3145_verify_holdout464_retry2048.jsonl",
]
POINTERS = {  # 공짜 greedy 포인터 (voter 아님/부분 겹침 무관 — 전부 독립 계보·교란)
    "fs1": "outputs/fewshot3_h3145_holdout464.jsonl",
    "fs2": "outputs/fewshot_set2_h3145_holdout464.jsonl",
    "verbose": "outputs/verbose_distill_holdout464_retry2048.jsonl",
    "grpo96": "outputs/grpo_3145_passrate94_steps96_holdout464_retry2048.jsonl",
    "grpo24": "outputs/grpo_3145_passrate94_steps24_holdout464_retry2048.jsonl",
    "ck100": "outputs/grpo_scaleup_ck100_holdout464.jsonl",
    "ck200": "outputs/grpo_scaleup_ck200_holdout464.jsonl",
    "ck250": "outputs/grpo_scaleup_ck250_holdout464.jsonl",
}
EXTRA_POINTERS = {  # 생성 완료 시 자동 포함
    "fs1shot": "outputs/fewshot_1shot_h3145_holdout464.jsonl",
    "fs6shot": "outputs/fewshot_6shot_h3145_holdout464.jsonl",
    "basefs1": "outputs/fewshot_set1_base_holdout464.jsonl",
}
FS1_8 = "outputs/fewshot3_h3145_n8_holdout464_seed20260925.jsonl"

gold = {r["id"]: S(norm(r["answer"])) for r in csv.DictReader(
    open(ROOT / "data/holdout/official_holdout_464_clean.csv", encoding="utf-8-sig"))}
ids = [i for i in gold if gold[i] is not None]
calib = {i for i in ids if int(hashlib.sha256(i.encode()).hexdigest(), 16) % 2 == 0}

vp = []
for rel in VOTERS:
    d = {}
    for r in jl(rel):
        p = r.get("prediction")
        d[r["id"]] = S(norm(p) if p is not None else norm(extract_v2(r.get("response"))))
    vp.append(d)
support = {}
for i in ids:
    c = Counter(v for v in (d.get(i) for d in vp) if v is not None)
    support[i] = c.most_common(1)[0][1] if c else 0
final660 = {r["id"]: S(norm(r.get("prediction"))) for r in jl("outputs/champion660_holdout464_equivalent.jsonl")}
ck_old = {r["id"]: [S(norm(x)) for x in r["predictions"]] for r in jl("outputs/ck150_n8_holdout464_seed20260920.jsonl")}
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
        if m is not None and m != base665[i] and guard_ok(i, base665[i], m):
            base665[i] = m

# pool16 (+가중치)
pool16, pw = defaultdict(list), defaultdict(lambda: defaultdict(float))
for rel in ["outputs/ck150_n8lp_holdout464_seed20260924.jsonl",
            "outputs/h3145_n8lp_holdout464_seed20260924.jsonl"]:
    for r in jl(rel):
        for p, cf in zip(r["predictions"], r["confidence"]):
            p = S(norm(p)) if p is not None else None
            pool16[r["id"]].append(p)
            if p is not None and cf.get("min_group") is not None:
                pw[r["id"]][p] += math.exp(cf["min_group"] * 4)


def votes(i, a):
    return sum(1 for p in pool16.get(i, []) if p == a)


fs1g = {r["id"]: S(r["prediction"]) for r in jl(POINTERS["fs1"])}
fs1_8 = {r["id"]: [S(p) for p in r.get("predictions", [])] for r in jl(FS1_8)}

# 세트1 규칙 재적용 -> 672 등가
s1 = []
for i in ids:
    a = fs1g.get(i)
    if a is None or a == base665[i]:
        continue
    if votes(i, a) < 4 or votes(i, a) <= votes(i, base665[i]):
        continue
    if sum(1 for p in fs1_8.get(i, []) if p == a) < 2:
        continue
    s1.append((i, a))
base672 = dict(base665)
for i, a in s1:
    base672[i] = a
B = sum(1 for i in ids if base672[i] == gold[i])
print(f"672 등가 = {B}/464 (세트1 플립 {len(s1)}개)  [타입 수정판]")
s1ids = {i for i, _ in s1}


def judge(tag, flips, show=False):
    g = sum(1 for i, a in flips if a == gold[i])
    r = sum(1 for i, a in flips if base672[i] == gold[i])
    dc = sum(1 for i, a in flips if i in calib and a == gold[i]) - \
         sum(1 for i, a in flips if i in calib and base672[i] == gold[i])
    print(f"{tag:<52} 교체 {len(flips):>3} g{g}/r{r} 델타 {g-r:+d} (cal{dc:+d}/val{(g-r)-dc:+d})")
    if show:
        for i, a in flips:
            m = "O" if a == gold[i] else ("X<-O" if base672[i] == gold[i] else "X<-X")
            print(f"    {i}: {base672[i]}->{a} {m} 표{votes(i,a)}vs{votes(i,base672[i])}")
    return g - r


# ── R0: 무포인터 상대우위 ──
print("\n[R0] 무포인터: pool16 최빈 a != 챔피언, 표>=k, 표(a) >= 표(champ)+마진")
for k in (4, 5, 6):
    for margin in (1, 2, 3, 4):
        flips = []
        for i in ids:
            m, nv = umode(pool16.get(i, []))
            if m is None or m == base672[i] or nv < k:
                continue
            if nv < votes(i, base672[i]) + margin:
                continue
            flips.append((i, m))
        judge(f"R0 k={k} margin>={margin}", flips)

# ── R1: 다중 포인터 합의 ──
ptr = {}
for name, rel in {**POINTERS, **EXTRA_POINTERS}.items():
    if not (ROOT / rel).exists():
        continue
    d = {}
    for r in jl(rel):
        p = r.get("prediction")
        d[r["id"]] = S(norm(p) if p is not None else norm(extract_v2(r.get("response"))))
    ptr[name] = d
print(f"\n[R1] 다중 포인터 합의 (보유 {len(ptr)}종: {', '.join(ptr)})")
for P in (2, 3, 4):
    for k in (3, 4, 5):
        flips = []
        for i in ids:
            cnt = Counter(d.get(i) for d in ptr.values() if d.get(i) is not None and d.get(i) != base672[i])
            if not cnt:
                continue
            a, n = cnt.most_common(1)[0]
            if n < P or votes(i, a) < k or votes(i, a) <= votes(i, base672[i]):
                continue
            flips.append((i, a))
        judge(f"R1 포인터합의>={P} & 표>={k} & 상대우위", flips)

# ── R2: 확신 가중 마진 (경합 구간용) ──
print("\n[R2] 확신 가중: W(a) > ratio x W(champ), raw 표>=4 (포인터 = fs1+fs2 합집합)")
for ratio in (1.5, 2.0, 3.0):
    flips = []
    for i in ids:
        for g_ in (fs1g.get(i), ptr.get("fs2", {}).get(i)):
            if g_ is None or g_ == base672[i]:
                continue
            if votes(i, g_) < 4:
                continue
            if pw[i].get(g_, 0) <= ratio * pw[i].get(base672[i], 0):
                continue
            flips.append((i, g_))
            break
    judge(f"R2 W비율>{ratio}", flips)
