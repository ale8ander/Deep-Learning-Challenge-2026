"""holdout464 위에서 '현행 제출본'과 대안 규칙들을 **같은 기준으로** 비교한다.

왜 이 스크립트가 필요한가
------------------------
2026-08-28 세션까지 이 프로젝트는 후보 규칙을 `hybrid_3145` 대비로 재고 x1.79(셋 크기)
x0.5(챔피언 보정)로 환산해 제출해 왔다. 그 환산이 틀려서 제출이 세 번 연속 -1 이 났다.

이제 voter 5종이 전부 갖춰져 챔피언을 holdout464 에서 직접 채점할 수 있다(365/464).
실측된 환산 오차는 이렇다:

    챔피언 - hybrid_3145 : holdout464 에서 +8, Public 831 에서 +29
    -> x1.79 환산이면 +14 여야 하는데 실제는 +29. **holdout464 는 챔피언 우위를 절반으로 과소평가한다.**

그러므로 앞으로 모든 판정은 hybrid_3145 가 아니라 **이 스크립트가 만드는 '현행 제출본
등가물'** 대비 gain/regression 으로 한다.

비교 대상
--------
  base_champion  : 5-voter + support4 SC override                     (Public 623)
  current_sub    : base_champion + TIR 표수<=3 mc2 + 표수4~5 mc4       (Public 647)
  risky_sub      : base_champion + TIR risky>=1 mc1 (양 구간)          (Public 646, 기각됨)
  lineage371     : 3계보 24샘플 mc11 (챔피언을 대체)                    (미제출)
  lineage371+TIR : lineage371 위에 현행 TIR override 를 얹은 것         (미제출)

⚠️ 엔진 혼재 주의: voter 중 hybrid_4145 / verify 는 5090+vLLM 산출물이고 나머지 3개는
A100+HF 산출물이다. `outputs/enginecheck_*` 로 일치율을 확인할 것.
"""
import csv
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402

GOLD = ROOT / "data/holdout/official_holdout_464_clean.csv"
VOTERS = [
    ("hybrid_3145", "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl"),
    ("hybrid_3244", "outputs/hybrid_3244_holdout464_retry2048.jsonl"),
    ("external_3000", "outputs/external_3000_holdout464_retry2048.jsonl"),
    ("hybrid_4145", "outputs/hybrid_4145_holdout464_retry2048.jsonl"),
    ("verify", "outputs/hybrid_3145_verify_holdout464_retry2048.jsonl"),
]
SC_MAIN = "outputs/self_consistency_confidence_n8_holdout464.jsonl"
SC_LINEAGES = {
    "h3145": "outputs/self_consistency_confidence_n8_holdout464.jsonl",
    "grpo96": "outputs/self_consistency_grpo96_n8_holdout464_seed20260826.jsonl",
    "verbose": "outputs/self_consistency_verbose_n8_holdout464_seed20260827.jsonl",
}
TIR_VOTE3 = "outputs/tir_sc8_holdout464_vote3_to60.jsonl"
TIR_VOTE45 = "outputs/tirc_hybrid3145_holdout464_vote45.jsonl"

INT_FINAL = re.compile(
    r"(?:final answer|정답)\s*(?:is|:|=)?\s*\$?\\?\(?\s*(-?\d[\d,]*)\s*\\?\)?\$?\s*\.?\s*$",
    re.I | re.M)
BOXED_INT = re.compile(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}")


def risky(t):
    if not t:
        return True
    tail = t[-400:]
    return not (INT_FINAL.search(tail) or BOXED_INT.search(tail))


def jl(path):
    for line in open(ROOT / path, encoding="utf-8"):
        if line.strip():
            yield json.loads(line)


def plur(counter, mc):
    top = counter.most_common()
    if top and top[0][1] >= mc and not (len(top) > 1 and top[0][1] == top[1][1]):
        return top[0][0]
    return None


def main():
    gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(open(GOLD, encoding="utf-8-sig"))}
    ids = [i for i in gold if gold[i] is not None]

    # --- voter 예측 ---
    vp = []
    for name, rel in VOTERS:
        d = {}
        for r in jl(rel):
            p = r.get("prediction")
            d[r["id"]] = norm(p) if p is not None else norm(extract_v2(r.get("response")))
        vp.append((name, d))

    # --- SC 풀 (hybrid3145 N=8): 표수, risky, baseline ---
    sc, samples = {}, {}
    for r in jl(SC_MAIN):
        resp = r.get("responses") or []
        preds = [norm(extract_v2(t)) for t in resp] if resp else \
                [norm(x) for x in (r.get("sample_predictions") or [])]
        samples[r["id"]] = preds
        c = Counter(p for p in preds if p is not None)
        sc[r["id"]] = {"votes": c.most_common(1)[0][1] if c else 0,
                       "nrisky": sum(risky(t) for t in resp) if resp else 0}

    # --- 5-voter + support4 override = 챔피언 ---
    champ, support = {}, {}
    for i in ids:
        votes = [d.get(i) for _, d in vp]
        c = Counter(v for v in votes if v is not None)
        support[i] = c.most_common(1)[0][1] if c else 0
        winners = [a for a, n in c.items() if n == max(c.values())] if c else []
        champ[i] = winners[0] if (len(winners) == 1 and max(c.values()) >= 2) else votes[0]
    for i in ids:
        if support[i] == 4:
            p = plur(Counter(x for x in samples.get(i, []) if x is not None), 4)
            if p is not None:
                champ[i] = p

    # --- 3계보 24샘플 mc11 ---
    pool24 = {i: [] for i in ids}
    for rel in SC_LINEAGES.values():
        for r in jl(rel):
            if r["id"] not in pool24:
                continue
            resp = r.get("responses")
            pool24[r["id"]] += ([norm(extract_v2(t)) for t in resp] if resp
                                else [norm(x) for x in (r.get("sample_predictions") or [])])
    lineage371 = {}
    for i in ids:
        p = plur(Counter(x for x in pool24[i] if x is not None), 11)
        lineage371[i] = p if p is not None else champ[i]

    # --- TIR override 레이어 ---
    def tir_counts(rel):
        d = {}
        for r in jl(rel):
            c = Counter()
            for k, v in (r.get("verified_counts") or {}).items():
                k = norm(k)
                if k is not None:
                    c[k] += v
            d[r["id"]] = c
        return d
    t3, t45 = tir_counts(TIR_VOTE3), tir_counts(TIR_VOTE45)

    def apply_tir(base, mode):
        out = dict(base)
        for i in ids:
            v = sc.get(i, {}).get("votes", 0)
            nr = sc.get(i, {}).get("nrisky", 0)
            if v <= 3 and i in t3:
                p = plur(t3[i], 1) if (mode == "risky" and nr >= 1) else (
                    plur(t3[i], 2) if mode == "current" else None)
                if p is not None:
                    out[i] = p
            elif 4 <= v <= 5 and i in t45:
                p = plur(t45[i], 1) if (mode == "risky" and nr >= 1) else (
                    plur(t45[i], 4) if mode == "current" else None)
                if p is not None:
                    out[i] = p
        return out

    cands = {
        "hybrid_3145 (기준선)": {i: vp[0][1].get(i) for i in ids},
        "base_champion (Public 623)": champ,
        "current_sub (Public 647)": apply_tir(champ, "current"),
        "risky_sub (Public 646, 기각)": apply_tir(champ, "risky"),
        "lineage371 (미제출)": lineage371,
        "lineage371 + 현행TIR": apply_tir(lineage371, "current"),
        "lineage371 + riskyTIR": apply_tir(lineage371, "risky"),
    }

    ref = cands["current_sub (Public 647)"]
    calib = {i for i in ids if int(hashlib.sha256(i.encode()).hexdigest(), 16) % 2 == 0}

    print(f"holdout464 정답 보유 {len(ids)}문제 "
          f"(calib {len(calib)} / valid {len(ids)-len(calib)})\n")
    print(f"{'후보':<30} {'점수':>9} {'vs 647본':>9} {'gain':>5} {'reg':>4} "
          f"{'변경':>5} {'calib':>6} {'valid':>6}")
    base_ref = sum(1 for i in ids if ref[i] == gold[i])
    for name, pred in cands.items():
        s = sum(1 for i in ids if pred[i] == gold[i])
        g = sum(1 for i in ids if pred[i] != ref[i] and pred[i] == gold[i])
        rg = sum(1 for i in ids if pred[i] != ref[i] and ref[i] == gold[i])
        ch = sum(1 for i in ids if pred[i] != ref[i])
        sc_ = sum(1 for i in ids if i in calib and pred[i] == gold[i]) - \
              sum(1 for i in ids if i in calib and ref[i] == gold[i])
        sv = (s - sum(1 for i in ids if i in calib and pred[i] == gold[i])) - \
             (base_ref - sum(1 for i in ids if i in calib and ref[i] == gold[i]))
        print(f"{name:<30} {s:>4}/{len(ids):<4} {s-base_ref:>+9} {g:>5} {rg:>4} "
              f"{ch:>5} {sc_:>+6} {sv:>+6}")

    print("\n(vs 647본 = 현행 최고 제출본과 같은 규칙을 holdout464 에 올린 것. "
          "이게 앞으로의 유일한 판정 기준이다.)")

    # 다른 스크립트가 '챔피언 대비'로 채점할 수 있게 기준선을 파일로 떨군다.
    # (C-1: 비교 기준은 항상 현행 제출본이다. hybrid_3145 대비로 재지 말 것.)
    out = ROOT / "outputs/champion_holdout464_equivalent.jsonl"
    with out.open("w", encoding="utf-8") as f:
        for i in ids:
            f.write(json.dumps({"id": i, "prediction": None if ref[i] is None else str(ref[i]),
                                "answer": None if gold[i] is None else str(gold[i])},
                               ensure_ascii=False) + "\n")
    print(f"기준선 저장: {out} ({len(ids)}행, {base_ref}/{len(ids)} 정답)")


if __name__ == "__main__":
    main()
