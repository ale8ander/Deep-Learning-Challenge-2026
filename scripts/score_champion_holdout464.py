"""현행 챔피언(5-voter + self-consistency support4 override)을 holdout464에서 채점한다.

이 프로젝트는 지금껏 챔피언을 Public 831에서만 측정해 왔고, 새 후보는 holdout464에서
측정해 왔다. 서로 다른 셋의 수치를 환산으로 비교하다 보니 채택/기각 판정이 흔들렸다.
이 스크립트는 두 규칙을 같은 464문제 위에 올려 직접 비교한다.

챔피언 규칙 (CONTEXT 요약표 기준):
  1. 5-voter plurality — hybrid_3145, hybrid_3244, external_3000, hybrid_4145,
     hybrid_3145(verify 프롬프트). 동률이거나 최다득표가 1표뿐이면 hybrid_3145로 fallback
     (scripts/ensemble_predictions.py choose_vote와 동일 규칙).
  2. support = 앙상블이 고른 답에 동의한 voter 수.
  3. support == 4 인 문제에만 hybrid_3145 N=8 self-consistency를 적용해,
     unique plurality가 min-count 4 이상이면 그 답으로 교체.
"""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/holdout/official_holdout_464_clean.csv"

VOTERS = [
    ("hybrid_3145", "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl"),
    ("hybrid_3244", "outputs/hybrid_3244_holdout464_retry2048.jsonl"),
    ("external_3000", "outputs/external_3000_holdout464_retry2048.jsonl"),
    ("hybrid_4145", "outputs/hybrid_4145_holdout464_retry2048.jsonl"),
    ("verify", "outputs/hybrid_3145_verify_holdout464_retry2048.jsonl"),
]
SC_3145 = "outputs/self_consistency_hybrid3145_n8_holdout500.jsonl"

NEW_POOLS = {
    "grpo96": "outputs/self_consistency_grpo96_n8_holdout464_seed20260826.jsonl",
    "h3145": SC_3145,
    "verbose": "outputs/self_consistency_verbose_n8_holdout464_seed20260827.jsonl",
}
GRPO_DET = "outputs/grpo_3145_passrate94_steps96_holdout464_retry2048.jsonl"


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


def load_preds(path):
    p = ROOT / path
    if not p.exists():
        return None
    out = {}
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = norm(d.get("prediction"))
    return out


def load_pool(path):
    p = ROOT / path
    if not p.exists():
        return None
    out = {}
    with open(p) as f:
        for line in f:
            d = json.loads(line)
            out[d["id"]] = [x for x in (norm(v) for v in d.get("sample_predictions", [])) if x is not None]
    return out


def choose(votes, fallback_idx=0):
    """ensemble_predictions.choose_vote와 동일: 단독 최다이고 2표 이상일 때만 채택."""
    counted = [v for v in votes if v is not None]
    if not counted:
        return None, 0
    c = Counter(counted)
    high = max(c.values())
    winners = [a for a, n in c.items() if n == high]
    if len(winners) == 1 and high >= 2:
        return winners[0], high
    fb = votes[fallback_idx]
    if fb is None:
        fb = next((v for v in votes if v is not None), None)
    return fb, sum(1 for v in votes if v == fb)


def main():
    with open(GOLD, newline="") as f:
        gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(f)}
    ids = list(gold)

    preds, missing = {}, []
    for name, path in VOTERS:
        p = load_preds(path)
        if p is None or len(set(ids) - set(p)) > 0:
            missing.append(name)
        else:
            preds[name] = p
    if missing:
        print(f"⏳ 아직 없는 voter: {', '.join(missing)} — 생성 완료 후 다시 실행")
        return

    print("== 개별 voter (holdout464) ==")
    for name, _ in VOTERS:
        print(f"  {name:16s} {sum(1 for i in ids if preds[name][i] == gold[i])}/464")

    order = [n for n, _ in VOTERS]
    ens, support = {}, {}
    for i in ids:
        v = [preds[n][i] for n in order]
        pick, sup = choose(v, 0)
        ens[i] = pick
        support[i] = sup
    base5 = sum(1 for i in ids if ens[i] == gold[i])
    print(f"\n== 5-voter plurality (tie=hybrid_3145): {base5}/464 ==")

    dist = Counter(support.values())
    print("  support 분포:", dict(sorted(dist.items())))
    for s in sorted(dist):
        sub = [i for i in ids if support[i] == s]
        print(f"    support={s}: {sum(1 for i in sub if ens[i]==gold[i])}/{len(sub)}")

    sc = load_pool(SC_3145)
    if sc is None:
        print("hybrid3145 N=8 풀 없음")
        return

    champ = dict(ens)
    changed = gain = loss = 0
    for i in ids:
        if support[i] != 4:
            continue
        c = Counter(sc.get(i, []))
        if not c:
            continue
        val, n = c.most_common(1)[0]
        tops = [v for v, k in c.items() if k == n]
        if n >= 4 and len(tops) == 1 and val != ens[i]:
            was = ens[i] == gold[i]
            now = val == gold[i]
            champ[i] = val
            changed += 1
            gain += now and not was
            loss += was and not now
    champ_score = sum(1 for i in ids if champ[i] == gold[i])
    print(
        f"\n== 챔피언 = 5-voter + support4 N=8 override(min4): {champ_score}/464 ==\n"
        f"   교체 {changed}문제, gain {gain} / regression {loss}"
    )

    # 새 규칙
    pools = {k: v for k, v in ((k, load_pool(p)) for k, p in NEW_POOLS.items()) if v}
    det = load_preds(GRPO_DET)
    if len(pools) == 3 and det:
        best = None
        for mc in range(4, 25):
            hit = 0
            for i in ids:
                s = []
                for n in pools:
                    s.extend(pools[n].get(i, []))
                c = Counter(s)
                pick = None
                if c:
                    val, k = c.most_common(1)[0]
                    tops = [v for v, q in c.items() if q == k]
                    if k >= mc and len(tops) == 1:
                        pick = val
                if pick is None:
                    pick = det.get(i)
                hit += pick == gold[i]
            if best is None or hit > best[1]:
                best = (mc, hit)
            if mc == 11:
                new11 = hit
        print(f"\n== 새 규칙 = 3계보 24샘플 (mc=11): {new11}/464 ==")
        print(f"   (전 구간 최고: mc={best[0]} → {best[1]}/464)")
        print(f"\n>>> 챔피언 {champ_score}  vs  새 규칙 {new11}  =>  {new11 - champ_score:+d}")


if __name__ == "__main__":
    main()
