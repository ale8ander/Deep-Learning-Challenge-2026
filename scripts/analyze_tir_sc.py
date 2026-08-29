"""TIR self-consistency 채점 — 코드검증된 답들의 다수결이 단일 실행보다 나은가.

비교 대상 (전부 holdout464의 표수<=3 서브셋 87문제, baseline = hybrid_3145 deterministic 17/87):
  - TIR greedy @768  : 게이트 발동 22, 순이득 +7 (gain 8 / reg 1), 정밀도 50%
  - TIR greedy @2048 : 게이트 발동 33, 순이득 +8 (gain 9 / reg 1), 정밀도 48%
  - 두 실행 합의     : 채택 17, 순이득 +8 (gain 8 / reg 0), 정밀도 65%
  - TIR SC N=8       : 이번에 재는 것

하네스가 문제별 verified_counts(코드검증된 답별 득표수)를 저장하므로
추론 재실행 없이 min-count를 스윕할 수 있다.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBSET = ROOT / "data/holdout/holdout464_vote3.csv"
BASE = ROOT / "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl"
SC = ROOT / "outputs/tir_sc8_holdout464_tirsft_fixed.jsonl"


def norm(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() in {"none", "inf", "-inf", "nan"}:
        return None
    try:
        return int(s)
    except ValueError:
        try:
            f = float(s)
        except ValueError:
            return None
        if f != f or f in (float("inf"), float("-inf")):
            return None
        try:
            return int(f)
        except (OverflowError, ValueError):
            return None


def main():
    with open(SUBSET, encoding="utf-8-sig", newline="") as f:
        gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(f)}

    base = {}
    for line in open(BASE):
        r = json.loads(line)
        if r["id"] in gold:
            base[r["id"]] = norm(r.get("prediction"))

    sc = {}
    for line in open(SC):
        r = json.loads(line)
        sc[r["id"]] = r

    ids = [i for i in sorted(gold) if i in sc and i in base]
    b_all = sum(1 for i in ids if base[i] == gold[i])
    calib = [i for i in ids if int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) % 2 == 0]
    valid = [i for i in ids if i not in set(calib)]

    print(f"holdout464 표수<=3 서브셋 {len(ids)}문제")
    print(f"  baseline(hybrid_3145) {b_all}/{len(ids)}")
    print(f"  calibration {len(calib)} / validation {len(valid)}")
    print()

    # 샘플 단위 통계
    total_samples = sum(len(sc[i]["sample_predictions"]) for i in ids)
    exec_ok = sum(1 for i in ids for s in sc[i]["sample_exec_status"] if s == "ok")
    code_any = sum(1 for i in ids for s in sc[i]["sample_exec_status"] if s is not None)
    print(f"=== 샘플 통계 (총 {total_samples}개) ===")
    print(f"  코드 생성 {code_any} ({code_any/total_samples:.0%})   실행 성공 {exec_ok} ({exec_ok/total_samples:.0%})")

    # 오라클: 8샘플 중 하나라도 정답인가 / 코드검증된 것 중 하나라도 정답인가
    orc_any = sum(1 for i in ids if gold[i] in {norm(p) for p in sc[i]["sample_predictions"]})
    orc_ver = sum(1 for i in ids if gold[i] in {norm(k) for k in sc[i]["verified_counts"]})
    print(f"  오라클(8샘플 중 하나라도 정답) {orc_any}/{len(ids)}")
    print(f"  오라클(코드검증된 답 중 정답 존재) {orc_ver}/{len(ids)}")
    print()

    print("=== min-count 스윕 ===")
    print(f"{'규칙':22s} {'점수':>8s} {'차이':>5s} {'gain':>5s} {'reg':>4s} {'채택':>5s} {'정밀도':>7s} {'calib':>6s} {'valid':>6s}")
    rows = []
    for mc in range(1, 9):
        pred, adopted, hit = {}, 0, 0
        for i in ids:
            counts = Counter({norm(k): v for k, v in sc[i]["verified_counts"].items() if norm(k) is not None})
            top = counts.most_common()
            if top and top[0][1] >= mc and not (len(top) > 1 and top[0][1] == top[1][1]):
                pred[i] = top[0][0]
                adopted += 1
                hit += int(pred[i] == gold[i])
            else:
                pred[i] = base[i]
        n = sum(1 for i in ids if pred[i] == gold[i])
        g = sum(1 for i in ids if base[i] != gold[i] and pred[i] == gold[i])
        rg = sum(1 for i in ids if base[i] == gold[i] and pred[i] != gold[i])
        nc = sum(1 for i in calib if pred[i] == gold[i]) - sum(1 for i in calib if base[i] == gold[i])
        nv = sum(1 for i in valid if pred[i] == gold[i]) - sum(1 for i in valid if base[i] == gold[i])
        prec = hit / adopted if adopted else 0
        print(f"{'min-count ' + str(mc):22s} {n:4d}/{len(ids)} {n-b_all:+5d} {g:5d} {rg:4d} {adopted:5d} {prec:6.0%} {nc:+6d} {nv:+6d}")
        rows.append((n, mc, g, rg, nc, nv, adopted, prec))

    best = max(rows)
    print()
    print(f"최고: min-count {best[1]} -> {best[0]}/{len(ids)} ({best[0]-b_all:+d}), "
          f"gain {best[2]} reg {best[3]}, 채택 {best[6]} 정밀도 {best[7]:.0%}, calib {best[4]:+d} valid {best[5]:+d}")
    print()
    print("채택 조건 (사전 고정: 순이득 +12 이상):")
    print(f"  순이득 +12 이상        {'통과' if best[0]-b_all >= 12 else f'미달 ({best[0]-b_all:+d})'}")
    print(f"  gain >= 2 x regression {'통과' if best[2] >= 2*best[3] else '실패'} ({best[2]} vs {best[3]})")
    print(f"  calibration 비음수      {'통과' if best[4] >= 0 else '실패'}")
    print(f"  validation 비음수       {'통과' if best[5] >= 0 else '실패'}")
    print()
    print("참고 — 기존 방식 (같은 87문제)")
    print("  TIR greedy @768   +7  (gain 8 / reg 1)  채택 22  정밀도 50%")
    print("  TIR greedy @2048  +8  (gain 9 / reg 1)  채택 33  정밀도 48%")
    print("  두 실행 합의      +8  (gain 8 / reg 0)  채택 17  정밀도 65%")


if __name__ == "__main__":
    main()
