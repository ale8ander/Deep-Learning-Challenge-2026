"""교차계보 합의를 선택 신호로 쓸 수 있는지 검사한다.

이 프로젝트의 1번 병목은 후보 생성이 아니라 선택이다(oracle 402 vs 실채택 367).
학습형 selector는 세 번 실패했고 min_count 통계 규칙만 재현됐다.
여기서는 학습 없이 "서로 독립 학습된 계보의 plurality가 일치하는가"라는
새 통계 신호가 정확도를 가르는지 본다.
"""
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/holdout/official_holdout_464_clean.csv"

POOLS = {
    "grpo96": "outputs/self_consistency_grpo96_n8_holdout464_seed20260826.jsonl",
    "h3145": "outputs/self_consistency_hybrid3145_n8_holdout500.jsonl",
    "verbose": "outputs/self_consistency_verbose_n8_holdout464_seed20260827.jsonl",
}
DET = {
    "grpo96": "outputs/grpo_3145_passrate94_steps96_holdout464_retry2048.jsonl",
    "h3145": "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl",
    "verbose": "outputs/verbose_distill_holdout464_retry2048.jsonl",
}


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
    with open(GOLD, newline="") as f:
        return {r["id"]: norm(r["answer"]) for r in csv.DictReader(f)}


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


def load_det(path):
    with open(ROOT / path) as f:
        return {json.loads(l)["id"]: norm(json.loads(l).get("prediction")) for l in open(ROOT / path)}


def plurality(samples):
    """(값, 표수, 동률여부)"""
    if not samples:
        return None, 0, False
    c = Counter(samples)
    val, n = c.most_common(1)[0]
    tops = [v for v, k in c.items() if k == n]
    return val, n, len(tops) > 1


def main():
    gold = load_gold()
    ids = list(gold)
    pools = {k: v for k, v in ((k, load_pool(p)) for k, p in POOLS.items()) if v is not None}
    dets = {k: load_det(p) for k, p in DET.items() if (ROOT / p).exists()}
    names = [n for n in ("grpo96", "h3145", "verbose") if n in pools]
    print("사용 가능한 계보:", ", ".join(names), "\n")
    if len(names) < 2:
        return

    # 각 계보의 plurality
    plur = {n: {i: plurality(pools[n].get(i, [])) for i in ids} for n in names}

    print("== 계보간 plurality 합의 여부별 정확도 ==")
    for a in range(len(names)):
        for b in range(a + 1, len(names)):
            na, nb = names[a], names[b]
            agree_hit = agree_n = dis_n = dis_hit_a = dis_hit_b = dis_either = 0
            for i in ids:
                va, ca, ta = plur[na][i]
                vb, cb, tb = plur[nb][i]
                if va is None or vb is None:
                    continue
                if va == vb:
                    agree_n += 1
                    agree_hit += va == gold[i]
                else:
                    dis_n += 1
                    dis_hit_a += va == gold[i]
                    dis_hit_b += vb == gold[i]
                    dis_either += (va == gold[i]) or (vb == gold[i])
            if agree_n:
                print(f"\n  [{na} vs {nb}]")
                print(f"    합의   {agree_n:3d}문제  정확도 {agree_hit}/{agree_n} = {agree_hit/agree_n:.1%}")
                print(
                    f"    불일치 {dis_n:3d}문제  {na} {dis_hit_a}  {nb} {dis_hit_b}  "
                    f"둘 중 하나라도 정답 {dis_either} ({dis_either/dis_n:.1%})"
                )
                print(
                    f"    → 불일치 구간에서 완벽히 고르면 +{dis_either - max(dis_hit_a, dis_hit_b)}문제 "
                    f"(현재 최선 단독 {max(dis_hit_a, dis_hit_b)})"
                )

    if len(names) >= 2:
        print("\n== 규칙: 계보 합의 시 채택, 불일치 시 fallback ==")
        for fb in names:
            for a in range(len(names)):
                for b in range(a + 1, len(names)):
                    na, nb = names[a], names[b]
                    if fb not in (na, nb):
                        continue
                    hit = 0
                    for i in ids:
                        va, _, _ = plur[na][i]
                        vb, _, _ = plur[nb][i]
                        pick = va if (va is not None and va == vb) else (
                            plur[fb][i][0] if plur[fb][i][0] is not None else dets.get(fb, {}).get(i)
                        )
                        hit += pick == gold[i]
                    print(f"  {na}+{nb} 합의, 불일치시 {fb:8s} → {hit}/464")

    print("\n== 합의 + 표수 결합 규칙 ==")
    if "grpo96" in pools and "h3145" in pools:
        for min_when_agree in (2, 3, 4):
            for min_when_alone in (5, 6, 7, 8):
                hit = 0
                for i in ids:
                    va, ca, ta = plur["grpo96"][i]
                    vb, cb, tb = plur["h3145"][i]
                    merged = pools["grpo96"].get(i, []) + pools["h3145"].get(i, [])
                    vm, cm, tm = plurality(merged)
                    if va is not None and va == vb and min(ca, cb) >= min_when_agree:
                        pick = va
                    elif vm is not None and not tm and cm >= min_when_alone:
                        pick = vm
                    else:
                        pick = dets.get("grpo96", {}).get(i)
                    hit += pick == gold[i]
                print(f"  합의(각 {min_when_agree}표+) 우선, 아니면 합산 plurality {min_when_alone}표+ → {hit}/464")


if __name__ == "__main__":
    main()
