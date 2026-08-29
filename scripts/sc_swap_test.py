"""SC 담당 교체 테스트 — 챔피언의 support4 SC override 샘플러(hybrid N=8)를
GRPO 체크포인트 N=8 로 바꿨을 때 660 등가(382/464)를 넘는지 실측.

범위 고정: 5-voter·밴드 정의(표수)·TIR 풀은 전부 기존 그대로, support4 override 에
쓰는 8샘플만 교체한다. TIR 가 이미 덮은 문제는 TIR 답을 유지한다(660 구조 보존) —
660 등가에서 최종답 == 챔피언623 답인 문제(=TIR 미발동/fallback)만 SC 교체의 영향을 받는다.

사용: /usr/bin/python3 scripts/sc_swap_test.py --ck-n8 outputs/ck150_n8_holdout464_seed20260920.jsonl
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

VOTERS = [
    "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl",
    "outputs/hybrid_3244_holdout464_retry2048.jsonl",
    "outputs/external_3000_holdout464_retry2048.jsonl",
    "outputs/hybrid_4145_holdout464_retry2048.jsonl",
    "outputs/hybrid_3145_verify_holdout464_retry2048.jsonl",
]
SC_MAIN = "outputs/self_consistency_confidence_n8_holdout464.jsonl"


def jl(rel):
    return [json.loads(l) for l in open(ROOT / rel) if l.strip()]


def champ623(sc_override_samples):
    """5-voter + support4 override. sc_override_samples: id -> [8 preds]"""
    vp = []
    for rel in VOTERS:
        d = {}
        for r in jl(rel):
            p = r.get("prediction")
            d[r["id"]] = norm(p) if p is not None else norm(extract_v2(r.get("response")))
        vp.append(d)
    out, support = {}, {}
    for i in ids:
        votes = [d.get(i) for d in vp]
        c = Counter(v for v in votes if v is not None)
        support[i] = c.most_common(1)[0][1] if c else 0
        winners = [a for a, n_ in c.items() if n_ == max(c.values())] if c else []
        out[i] = winners[0] if (len(winners) == 1 and max(c.values()) >= 2) else votes[0]
    for i in ids:
        if support[i] == 4:
            c = Counter(x for x in sc_override_samples.get(i, []) if x is not None)
            tp = c.most_common()
            if tp and tp[0][1] >= 4 and (len(tp) == 1 or tp[0][1] > tp[1][1]):
                out[i] = tp[0][0]
    return out, support


ap = argparse.ArgumentParser()
ap.add_argument("--ck-n8", required=True)
ap.add_argument("--label", default="ck150")
args = ap.parse_args()

gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(
    open(ROOT / "data/holdout/official_holdout_464_clean.csv", encoding="utf-8-sig"))}
ids = [i for i in gold if gold[i] is not None]

# hybrid SC 샘플 (기존 override 재료)
hyb_samples = {}
for r in jl(SC_MAIN):
    resp = r.get("responses") or []
    hyb_samples[r["id"]] = ([norm(extract_v2(t)) for t in resp] if resp
                            else [norm(x) for x in (r.get("sample_predictions") or [])])
# ck 샘플 (교체 재료)
ck_samples = {r["id"]: [norm(x) for x in r["predictions"]] for r in jl(args.ck_n8)}

base, support = champ623(hyb_samples)      # 기존 챔피언623
swapped, _ = champ623(ck_samples)          # SC 담당만 ck 로 교체한 챔피언623'

final660 = {}
for l in open(ROOT / "outputs/champion660_holdout464_equivalent.jsonl"):
    r = json.loads(l)
    final660[r["id"]] = norm(r.get("prediction"))

# TIR 미개입(최종==챔피언623) 문제에만 SC 교체 반영
new_final, applied = {}, 0
for i in ids:
    if swapped[i] != base[i] and final660[i] == base[i]:
        new_final[i] = swapped[i]
        applied += 1
    else:
        new_final[i] = final660[i]

s660 = sum(1 for i in ids if final660[i] == gold[i])
s623 = sum(1 for i in ids if base[i] == gold[i])
s623s = sum(1 for i in ids if swapped[i] == gold[i])
snew = sum(1 for i in ids if new_final[i] == gold[i])
g = sum(1 for i in ids if new_final[i] != final660[i] and new_final[i] == gold[i])
rg = sum(1 for i in ids if new_final[i] != final660[i] and final660[i] == gold[i])
n_s4 = sum(1 for i in ids if support[i] == 4)
print(f"support4 문제: {n_s4}개 | SC 교체로 답이 바뀐 문제(TIR 미개입만): {applied}개")
print(f"챔피언623 (hybrid SC): {s623}/464 → SC={args.label} 교체: {s623s}/464")
print(f"660 등가: {s660}/464 → SC={args.label} 교체 후: {snew}/464  ({snew-s660:+d}, gain {g}/reg {rg})")
print(f"판정 기준: 382 초과 + gain/reg 명확해야 채택 논의 (C-3: ±4 미만 노이즈)")
