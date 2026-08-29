"""Step 1 — 추가 추론 0. GRPO96 · verbose distill 두 계보의 **deterministic** 예측만으로
"합의하면 채택, 불일치하면 다른 신호로 선택"하는 단순 규칙들을 holdout464에서 채점한다.

왜 deterministic인가: 기존 371/464 규칙은 계보당 N=8 stochastic 샘플을 요구해 831 적용 시
추론 2~2.5시간이 든다. 만약 deterministic 합의 규칙만으로 비슷한 점수가 나오면 훨씬 싸다.

판정 기준은 GRPO96 deterministic 단독(360/464) 대비 gain/regression 분해다.
"""
import json
import csv
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/holdout/official_holdout_464_clean.csv"

DET_FILES = {
    "grpo96": "outputs/grpo_3145_passrate94_steps96_holdout464_retry2048.jsonl",
    "verbose": "outputs/verbose_distill_holdout464_retry2048.jsonl",
    "h3145": "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl",
    "h3244": "outputs/hybrid_3244_holdout464_retry2048.jsonl",
    "grpo24": "outputs/grpo_3145_passrate94_steps24_holdout464_retry2048.jsonl",
    # 5-voter 나머지 (생성 중이면 자동 skip)
    "external_3000": "outputs/external_3000_holdout464_retry2048.jsonl",
    "h4145": "outputs/hybrid_4145_holdout464_retry2048.jsonl",
    "verify": "outputs/hybrid_3145_verify_holdout464_retry2048.jsonl",
}

POOL_FILES = {
    "grpo96": "outputs/self_consistency_grpo96_n8_holdout464_seed20260826.jsonl",
    "verbose": "outputs/self_consistency_verbose_n8_holdout464_seed20260827.jsonl",
    "h3145": "outputs/self_consistency_hybrid3145_n8_holdout500.jsonl",
}

VOTER_ORDER = ["h3145", "h3244", "external_3000", "h4145", "verify"]


def norm(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


def load_gold():
    with open(GOLD, encoding="utf-8-sig", newline="") as f:
        return {r["id"]: norm(r["answer"]) for r in csv.DictReader(f)}


def load_det(path):
    p = ROOT / path
    if not p.exists():
        return None
    out = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            out[r["id"]] = norm(r.get("prediction"))
    return out


def load_pool(path):
    """문제별 sample_predictions 리스트(정수 정규화)를 돌려준다."""
    p = ROOT / path
    if not p.exists():
        return None
    out = {}
    with open(p) as f:
        for line in f:
            r = json.loads(line)
            samples = [norm(s) for s in r.get("sample_predictions", [])]
            out[r["id"]] = [s for s in samples if s is not None]
    return out


def plurality(counter):
    """최다득표 답. 동률이면 None."""
    if not counter:
        return None
    ranked = counter.most_common()
    if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
        return None
    return ranked[0][0]


def score(gold, ids, preds):
    return sum(1 for i in ids if preds.get(i) is not None and preds[i] == gold[i])


def decompose(gold, ids, base, cand):
    gain = [i for i in ids if base.get(i) != gold[i] and cand.get(i) == gold[i]]
    reg = [i for i in ids if base.get(i) == gold[i] and cand.get(i) != gold[i]]
    changed = [i for i in ids if base.get(i) != cand.get(i)]
    return gain, reg, changed


def main():
    gold = load_gold()
    ids = sorted(gold)

    det = {k: load_det(v) for k, v in DET_FILES.items()}
    missing = [k for k, v in det.items() if v is None]
    det = {k: v for k, v in det.items() if v is not None}
    pools = {k: load_pool(v) for k, v in POOL_FILES.items()}
    pools = {k: v for k, v in pools.items() if v is not None}

    print(f"holdout464 문제 수: {len(ids)}")
    print(f"deterministic 예측 로드: {sorted(det)}")
    if missing:
        print(f"  (아직 없음 — 생성 중이거나 미실행: {sorted(missing)})")
    print(f"sample pool 로드: {sorted(pools)}")
    print()

    print("=== 단독 deterministic 기준선 ===")
    for name in sorted(det):
        cov = sum(1 for i in ids if i in det[name])
        print(f"  {name:14s} {score(gold, ids, det[name]):3d}/464   (커버 {cov})")
    print()

    g, v = det["grpo96"], det["verbose"]
    agree = [i for i in ids if g.get(i) is not None and g[i] == v.get(i)]
    disagree = [i for i in ids if i not in set(agree)]

    print("=== GRPO96 vs verbose deterministic 합의 구조 ===")
    a_correct = sum(1 for i in agree if g[i] == gold[i])
    print(f"  합의   {len(agree):3d}문제 → {a_correct:3d} 정답 ({a_correct/max(len(agree),1):.1%})")
    d_g = sum(1 for i in disagree if g.get(i) == gold[i])
    d_v = sum(1 for i in disagree if v.get(i) == gold[i])
    d_either = sum(1 for i in disagree if g.get(i) == gold[i] or v.get(i) == gold[i])
    print(f"  불일치 {len(disagree):3d}문제 → grpo96 {d_g}, verbose {d_v}, 둘 중 하나라도 정답 {d_either} ({d_either/max(len(disagree),1):.1%})")
    print(f"  → 이 규칙군의 상한(불일치를 완벽히 고를 때): {a_correct + d_either}/464")
    print()

    # ---- 불일치 구간 tie-break 후보들 --------------------------------------
    tiebreaks = {}
    tiebreaks["grpo96(기준선)"] = lambda i: g.get(i)
    tiebreaks["verbose"] = lambda i: v.get(i)
    if "h3145" in det:
        tiebreaks["h3145"] = lambda i: det["h3145"].get(i)

    def det_majority(keys):
        def f(i):
            c = Counter(det[k][i] for k in keys if k in det and det[k].get(i) is not None)
            return plurality(c)
        return f

    avail4 = [k for k in ("grpo96", "verbose", "h3145", "h3244") if k in det]
    tiebreaks[f"det-majority{avail4}"] = det_majority(avail4)

    def pool_plur(keys, min_count=1):
        def f(i):
            c = Counter()
            for k in keys:
                if k in pools and i in pools[k]:
                    c.update(pools[k][i])
            p = plurality(c)
            if p is None or c[p] < min_count:
                return g.get(i)  # 동률/미달이면 기준선 유지
            return p
        return f

    for mc in (0, 4, 5, 6, 7, 8):
        keys = [k for k in ("grpo96", "verbose") if k in pools]
        if keys:
            tiebreaks[f"pool(grpo96+verbose) min{mc}"] = pool_plur(keys, mc)
    keys3 = [k for k in ("grpo96", "verbose", "h3145") if k in pools]
    if len(keys3) == 3:
        for mc in (0, 6, 8, 10, 11):
            tiebreaks[f"pool(3계보24) min{mc}"] = pool_plur(keys3, mc)

    # 5-voter support 기반 (voter 전부 있을 때만)
    if all(k in det for k in VOTER_ORDER):
        def five_voter(i):
            c = Counter(det[k][i] for k in VOTER_ORDER if det[k].get(i) is not None)
            if not c:
                return det["h3145"].get(i)
            ranked = c.most_common()
            if len(ranked) > 1 and ranked[0][1] == ranked[1][1]:
                return det["h3145"].get(i)
            if ranked[0][1] <= 1:
                return det["h3145"].get(i)
            return ranked[0][0]
        tiebreaks["5-voter plurality"] = five_voter

        def five_voter_support(i):
            """5-voter가 고른 답 + 그 support를 함께 본다: support>=3이면 그것, 아니면 기준선."""
            c = Counter(det[k][i] for k in VOTER_ORDER if det[k].get(i) is not None)
            ranked = c.most_common()
            if ranked and ranked[0][1] >= 3 and not (len(ranked) > 1 and ranked[0][1] == ranked[1][1]):
                return ranked[0][0]
            return g.get(i)
        tiebreaks["5-voter support>=3, else grpo96"] = five_voter_support

    print("=== 규칙: 합의하면 채택 / 불일치하면 아래 방식으로 선택 ===")
    print(f"{'tie-break 방식':42s} {'점수':>8s}  {'vs grpo96':>10s}  {'gain':>5s} {'reg':>4s} {'변경':>5s}")
    base = g
    base_score = score(gold, ids, base)
    rows = []
    for name, fn in tiebreaks.items():
        pred = {}
        for i in ids:
            pred[i] = g[i] if i in set(agree) else fn(i)
        s = score(gold, ids, pred)
        gain, reg, changed = decompose(gold, ids, base, pred)
        rows.append((s, name, len(gain), len(reg), len(changed)))
    for s, name, ng, nr, nc in sorted(rows, reverse=True):
        print(f"{name:42s} {s:4d}/464  {s-base_score:+10d}  {ng:5d} {nr:4d} {nc:5d}")
    print()
    print(f"참고 — GRPO96 deterministic 단독: {base_score}/464")
    print(f"참고 — 기록된 3계보 24샘플 min11 규칙: 371/464")


if __name__ == "__main__":
    main()
