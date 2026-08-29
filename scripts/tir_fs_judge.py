"""few-shot TIR 판정 — 672 등가(396/464) 기준.

새 우물 가설: few-shot 교란 TIR 가 기존 코드가 못 뚫던 문제에서 **코드 검증된** 답을
만든다. 확인 신호 = 실행 검증(vote 우위 아님 — 그 우물은 폐쇄 확정됨).

단계:
  0. 상한: base672 오답 68개 중 새 풀의 verified 최빈이 정답인 문제 수
  1. 규칙 스윕: 새 verified 최빈 a != 챔피언 & vc[a]>=k & vc[a] > vc[champ]+m
     (± 기존 4풀 codeguard 반대 없음 / ± sample 표 보조)
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


def S(x):
    return None if x is None else str(x)


def jl(rel):
    return [json.loads(l) for l in open(ROOT / rel) if l.strip()]


# ── 672 등가 재구성 (pointer_lab 과 동일 로직 요약) ──
VOTERS = [
    "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl",
    "outputs/hybrid_3244_holdout464_retry2048.jsonl",
    "outputs/external_3000_holdout464_retry2048.jsonl",
    "outputs/hybrid_4145_holdout464_retry2048.jsonl",
    "outputs/hybrid_3145_verify_holdout464_retry2048.jsonl",
]
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
ver_old = {}
for rel in ["outputs/tir_sc8_holdout464_vote3_to60.jsonl",
            "outputs/tir_repair1_holdout464_vote3.jsonl",
            "outputs/tir_repair_nocode_holdout464_vote3.jsonl",
            "outputs/tirc_hybrid3145_holdout464_vote45.jsonl"]:
    for r in jl(rel):
        c = ver_old.setdefault(r["id"], Counter())
        for a, v in (r.get("verified_counts") or {}).items():
            a = tnorm(a)
            if a is not None:
                c[a] += v


def umode(preds, min_v=1):
    tp = Counter(x for x in preds if x is not None).most_common()
    if tp and not (len(tp) > 1 and tp[0][1] == tp[1][1]) and tp[0][1] >= min_v:
        return tp[0][0], tp[0][1]
    return None, 0


def old_guard_ok(i, cur, mode):
    vc = ver_old.get(i, Counter())
    return not (vc and vc.get(tnorm(str(cur)), 0) > vc.get(tnorm(str(mode)), 0))


base665 = dict(final660)
for i in ids:
    if support[i] <= 4:
        m, nv = umode(ck_old.get(i, []), 5)
        if m is not None and m != base665[i] and old_guard_ok(i, base665[i], m):
            base665[i] = m
pool16 = {}
for rel in ["outputs/ck150_n8lp_holdout464_seed20260924.jsonl",
            "outputs/h3145_n8lp_holdout464_seed20260924.jsonl"]:
    for r in jl(rel):
        pool16.setdefault(r["id"], []).extend(S(norm(p)) if p is not None else None
                                              for p in r["predictions"])


def votes(i, a):
    return sum(1 for p in pool16.get(i, []) if p == a)


fs1g = {r["id"]: S(r["prediction"]) for r in jl("outputs/fewshot3_h3145_holdout464.jsonl")}
fs1_8 = {r["id"]: [S(p) for p in r.get("predictions", [])]
         for r in jl("outputs/fewshot3_h3145_n8_holdout464_seed20260925.jsonl")}
base672 = dict(base665)
for i in ids:
    a = fs1g.get(i)
    if a is None or a == base672[i]:
        continue
    if votes(i, a) < 4 or votes(i, a) <= votes(i, base672[i]):
        continue
    if sum(1 for p in fs1_8.get(i, []) if p == a) < 2:
        continue
    base672[i] = a
B = sum(1 for i in ids if base672[i] == gold[i])
wrong = {i for i in ids if base672[i] != gold[i]}
print(f"672 등가 = {B}/464, 오답 {len(wrong)}")

# ── 새 few-shot TIR 풀 ──
new = {}
for r in jl("outputs/tir_fs8_holdout464_seed20260930.jsonl"):
    c = Counter()
    for a, v in (r.get("verified_counts") or {}).items():
        a = tnorm(a)
        if a is not None:
            c[S(a)] += v
    new[r["id"]] = {"vc": c, "preds": [S(norm(p)) for p in r.get("sample_predictions", [])]}

cov = sum(1 for i in ids if i in new)
orc = sum(1 for i in wrong if i in new and new[i]["vc"] and
          max(new[i]["vc"], key=new[i]["vc"].get) == gold[i])
orc_any = sum(1 for i in wrong if i in new and new[i]["vc"].get(gold[i], 0) >= 1)
print(f"커버리지 {cov}/464 | 오답 68 중 새 verified 최빈=정답 {orc} | verified>=1 {orc_any}")


def judge(tag, flips):
    g = sum(1 for i, a in flips if a == gold[i])
    r = sum(1 for i, a in flips if base672[i] == gold[i])
    dc = sum(1 for i, a in flips if i in calib and a == gold[i]) - \
         sum(1 for i, a in flips if i in calib and base672[i] == gold[i])
    print(f"{tag:<56} 교체 {len(flips):>3} g{g}/r{r} 델타 {g-r:+d} (cal{dc:+d}/val{(g-r)-dc:+d})")
    return flips


print("\n[T] verified 최빈 a != 챔피언 & vc[a]>=k & vc[a] >= vc[champ]+m")
for k in (2, 3, 4, 5):
    for m in (1, 2, 3):
        flips = []
        for i in ids:
            if i not in new or not new[i]["vc"]:
                continue
            tp = new[i]["vc"].most_common()
            if len(tp) > 1 and tp[0][1] == tp[1][1]:
                continue
            a, n = tp[0]
            if a == base672[i] or n < k:
                continue
            if n < new[i]["vc"].get(base672[i], 0) + m:
                continue
            flips.append((i, a))
        judge(f"T k={k} m={m}", flips)

print("\n[T+G] 위에 기존 4풀 codeguard(반대 없음) 추가")
for k in (3, 4, 5):
    for m in (2, 3):
        flips = []
        for i in ids:
            if i not in new or not new[i]["vc"]:
                continue
            tp = new[i]["vc"].most_common()
            if len(tp) > 1 and tp[0][1] == tp[1][1]:
                continue
            a, n = tp[0]
            if a == base672[i] or n < k:
                continue
            if n < new[i]["vc"].get(base672[i], 0) + m:
                continue
            if not old_guard_ok(i, base672[i], a):
                continue
            flips.append((i, a))
        judge(f"T+G k={k} m={m}", flips)
