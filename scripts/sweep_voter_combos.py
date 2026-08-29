"""5-voter 구성 전수 스윕 — holdout464, 665 등가 체인 기준 (CPU-only).

12개 후보(현행 5 + GRPO24/96 + ck100/150/200/250 + verbose)에서 5개를 뽑는
모든 조합 x fallback(동률심판) 선택에 대해 배포 체인 전체를 재구성해 채점한다.

체인 (배포 665 와 동일 구조):
  1. 5-voter 다수결 (>=2표 유일 최빈, 아니면 fallback voter 답)
  2. support==4 -> hybrid SC N=8 mc4 유일최빈 override
  3. TIR 레이어: 배포 체인에서 TIR 이 발동한 문제(final660 != 배포 base623)는
     final660 답 유지 (밴드가 SC 표수 기준이라 voter 와 독립)
  4. 삼중 게이트: support<=4 x ck150 N=8 유일최빈>=5표 x 코드가드(TIR 4풀 합산
     verified_counts 에서 기존 답 표 > 새 답 표이면 취소)

판정: 배포 조합의 체인 결과(665 등가)를 기준으로 gain/reg/calib/valid.
⚠️ 792조합 x5 fallback 다중비교 — 선택효과만으로 +5~7 나온다. 발사 바는
델타 >= +8 & 변경 >= 15 & calib/valid 양수, 통과해도 메가2000 확인 사격 필수.
"""
import csv
import hashlib
import json
import sys
from collections import Counter
from itertools import combinations
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402
from tir_common import normalize as tnorm  # noqa: E402

CANDIDATES = [
    ("h3145",   "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl"),
    ("h3244",   "outputs/hybrid_3244_holdout464_retry2048.jsonl"),
    ("ext3000", "outputs/external_3000_holdout464_retry2048.jsonl"),
    ("h4145",   "outputs/hybrid_4145_holdout464_retry2048.jsonl"),
    ("verify",  "outputs/hybrid_3145_verify_holdout464_retry2048.jsonl"),
    ("grpo24",  "outputs/grpo_3145_passrate94_steps24_holdout464_retry2048.jsonl"),
    ("grpo96",  "outputs/grpo_3145_passrate94_steps96_holdout464_retry2048.jsonl"),
    ("ck100",   "outputs/grpo_scaleup_ck100_holdout464.jsonl"),
    ("ck150",   "outputs/grpo_scaleup_ck150_holdout464.jsonl"),
    ("ck200",   "outputs/grpo_scaleup_ck200_holdout464.jsonl"),
    ("ck250",   "outputs/grpo_scaleup_ck250_holdout464.jsonl"),
    ("verbose", "outputs/verbose_distill_holdout464_retry2048.jsonl"),
]
DEPLOYED = ("h3145", "h3244", "ext3000", "h4145", "verify")  # fallback = h3145
SC_MAIN = "outputs/self_consistency_confidence_n8_holdout464.jsonl"
CK_N8 = "outputs/ck150_n8_holdout464_seed20260920.jsonl"
FINAL660 = "outputs/champion660_holdout464_equivalent.jsonl"
TIR_POOLS = [  # 코드가드용 — 831 빌더의 4풀 홀드아웃 대응물
    "outputs/tir_sc8_holdout464_vote3_to60.jsonl",
    "outputs/tir_repair1_holdout464_vote3.jsonl",
    "outputs/tir_repair_nocode_holdout464_vote3.jsonl",
    "outputs/tirc_hybrid3145_holdout464_vote45.jsonl",
]


def jl(rel):
    return [json.loads(l) for l in open(ROOT / rel) if l.strip()]


gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(
    open(ROOT / "data/holdout/official_holdout_464_clean.csv", encoding="utf-8-sig"))}
ids = [i for i in gold if gold[i] is not None]
calib = {i for i in ids if int(hashlib.sha256(i.encode()).hexdigest(), 16) % 2 == 0}

preds = {}
for name, rel in CANDIDATES:
    d = {}
    for r in jl(rel):
        p = r.get("prediction")
        d[r["id"]] = norm(p) if p is not None else norm(extract_v2(r.get("response")))
    preds[name] = d
    miss = [i for i in ids if i not in d]
    if miss:
        sys.exit(f"{name}: holdout464 커버리지 부족 {len(miss)}개 (예 {miss[:3]})")

hyb_samples = {}
for r in jl(SC_MAIN):
    resp = r.get("responses") or []
    hyb_samples[r["id"]] = ([norm(extract_v2(t)) for t in resp] if resp
                            else [norm(x) for x in (r.get("sample_predictions") or [])])

ck_mode = {}  # id -> (mode, votes) 유일최빈만
for r in jl(CK_N8):
    c = Counter(norm(x) for x in r["predictions"] if norm(x) is not None)
    tp = c.most_common()
    if tp and not (len(tp) > 1 and tp[0][1] == tp[1][1]):
        ck_mode[r["id"]] = (tp[0][0], tp[0][1])

ver = {}
for rel in TIR_POOLS:
    for r in jl(rel):
        c = ver.setdefault(r["id"], Counter())
        for a, v in (r.get("verified_counts") or {}).items():
            a = tnorm(a)
            if a is not None:
                c[a] += v

final660 = {r["id"]: norm(r.get("prediction")) for r in jl(FINAL660)}


def chain(combo, fb):
    """combo: voter 이름 튜플, fb: fallback voter 이름. return (final dict, support dict)"""
    out, support = {}, {}
    dicts = [preds[n] for n in combo]
    fbd = preds[fb]
    for i in ids:
        votes = [d.get(i) for d in dicts]
        c = Counter(v for v in votes if v is not None)
        support[i] = c.most_common(1)[0][1] if c else 0
        winners = [a for a, n_ in c.items() if n_ == max(c.values())] if c else []
        out[i] = winners[0] if (len(winners) == 1 and max(c.values()) >= 2) else fbd.get(i)
        if support[i] == 4:  # support4 SC override
            sc = Counter(x for x in hyb_samples.get(i, []) if x is not None)
            tp = sc.most_common()
            if tp and tp[0][1] >= 4 and (len(tp) == 1 or tp[0][1] > tp[1][1]):
                out[i] = tp[0][0]
        if i in tir_fired:  # TIR 레이어 (voter 독립)
            out[i] = final660[i]
        # 삼중 게이트
        if support[i] <= 4 and i in ck_mode:
            mode, nv = ck_mode[i]
            if nv >= 5 and str(mode) != str(out[i]):
                vc = ver.get(i, Counter())
                if not (vc and vc.get(tnorm(str(out[i])), 0) > vc.get(tnorm(str(mode)), 0)):
                    out[i] = mode
    return out, support


def score(pred):
    return sum(1 for i in ids if pred[i] == gold[i])


# --- 배포 base623 재구성 -> TIR 발동 집합 ---
_b623, _sup = {}, {}
dicts = [preds[n] for n in DEPLOYED]
for i in ids:
    votes = [d.get(i) for d in dicts]
    c = Counter(v for v in votes if v is not None)
    _sup[i] = c.most_common(1)[0][1] if c else 0
    winners = [a for a, n_ in c.items() if n_ == max(c.values())] if c else []
    _b623[i] = winners[0] if (len(winners) == 1 and max(c.values()) >= 2) else votes[0]
    if _sup[i] == 4:
        sc = Counter(x for x in hyb_samples.get(i, []) if x is not None)
        tp = sc.most_common()
        if tp and tp[0][1] >= 4 and (len(tp) == 1 or tp[0][1] > tp[1][1]):
            _b623[i] = tp[0][0]
tir_fired = {i for i in ids if final660.get(i) != _b623[i]}
print(f"sanity: base623 = {score(_b623)}/464 (기대 365), "
      f"660등가 = {score(final660)}/464 (기대 382), TIR 발동 {len(tir_fired)}개")

base_pred, _ = chain(DEPLOYED, "h3145")
BASE = score(base_pred)
print(f"sanity: 배포 조합 체인(665 등가) = {BASE}/464 (기대 ~387)\n")

# --- 전수 스윕 ---
rows = []
names = [n for n, _ in CANDIDATES]
for combo in combinations(names, 5):
    for fb in combo:
        pred, _ = chain(combo, fb)
        s = score(pred)
        if s < BASE:      # 기준 미만은 상세 계산 생략
            rows.append((s, combo, fb, None))
            continue
        g = sum(1 for i in ids if pred[i] != base_pred[i] and pred[i] == gold[i])
        rg = sum(1 for i in ids if pred[i] != base_pred[i] and base_pred[i] == gold[i])
        ch = sum(1 for i in ids if pred[i] != base_pred[i])
        dc = sum(1 for i in ids if i in calib and pred[i] == gold[i]) - \
             sum(1 for i in ids if i in calib and base_pred[i] == gold[i])
        dv = (s - BASE) - dc
        rows.append((s, combo, fb, (g, rg, ch, dc, dv)))

rows.sort(key=lambda r: -r[0])
n_above = sum(1 for r in rows if r[0] > BASE)
print(f"조합 x fallback 총 {len(rows)}개 | 기준 {BASE} 초과 {n_above}개\n")
print(f"{'조합':<44} {'fb':<8} {'점수':>7} {'델타':>5} {'g/r':>7} {'변경':>4} {'cal':>4} {'val':>4}")
for s, combo, fb, d in rows[:25]:
    tag = "+".join(combo)
    if d:
        g, rg, ch, dc, dv = d
        print(f"{tag:<44} {fb:<8} {s:>4}/464 {s-BASE:>+5} {g:>3}/{rg:<3} {ch:>4} {dc:>+4} {dv:>+4}")
    else:
        print(f"{tag:<44} {fb:<8} {s:>4}/464 {s-BASE:>+5}")

# 배포 조합의 순위
dep_scores = [s for s, c, fb, _ in rows if c == DEPLOYED and fb == "h3145"]
better = sum(1 for r in rows if r[0] > dep_scores[0])
print(f"\n배포 조합 점수 {dep_scores[0]}, 이보다 높은 조합 {better}개")
print("발사 바: 델타>=+8 & 변경>=15 & calib/valid 양수 -> 통과 시에도 메가2000 확인 필수")
