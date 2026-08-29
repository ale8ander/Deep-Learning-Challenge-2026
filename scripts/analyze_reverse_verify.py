"""역검증 결과로 후보를 고르는 규칙들을 채점한다.

비교 대상은 세 가지다.
  1. baseline (hybrid_3145 deterministic)
  2. 현행 배포 규칙 (TIR 코드검증 plurality, min-count 2)
  3. 이번 세션의 risky 게이트 (risky>=1 + min-count 1)
그 위에 역검증 규칙들을 얹어 순이득/gain/regression 을 본다.

모든 규칙은 id 해시 반반으로 calibration/validation 분할검증을 함께 출력한다
(scripts/validate_sc_rule_split.py 와 같은 분할). threshold 를 여러 개 훑는 규칙은
calibration 에서 고른 값이 validation 에서도 살아남는지 봐야 채택할 수 있다.
"""
import argparse
import hashlib
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402

INT_FINAL = re.compile(
    r"(?:final answer|정답)\s*(?:is|:|=)?\s*\$?\\?\(?\s*(-?\d[\d,]*)\s*\\?\)?\$?\s*\.?\s*$",
    re.I | re.M)
BOXED_INT = re.compile(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}")

LINEAGES = {
    "hybrid3145": "outputs/tir_sc8_holdout464_vote3_to60.jsonl",
    "tirc_3145": "outputs/tirc_hybrid3145_holdout464_vote3.jsonl",
    "grpo96": "outputs/tirc_grpo96_holdout464_vote3.jsonl",
    "tirsft": "outputs/tirc_tirsft_holdout464_vote3.jsonl",
    "tirexec": "outputs/tirc_tirexec_holdout464_vote3.jsonl",
}
SC_POOL = "outputs/self_consistency_confidence_n8_holdout464.jsonl"


def risky(text):
    if not text:
        return True
    tail = text[-400:]
    return not (INT_FINAL.search(tail) or BOXED_INT.search(tail))


def load_state():
    sc = {}
    for line in open(ROOT / SC_POOL):
        r = json.loads(line)
        sc[r["id"]] = {
            "base": norm(r["baseline_prediction"]),
            "ans": norm(r["answer"]),
            "nrisky": sum(risky(t) for t in r["responses"]),
        }
    pools = {}
    for name, rel in LINEAGES.items():
        p = ROOT / rel
        if not p.exists():
            continue
        d = {}
        for line in open(p):
            r = json.loads(line)
            c = Counter()
            for k, v in (r.get("verified_counts") or {}).items():
                k = norm(k)
                if k is not None:
                    c[k] += v
            d[r["id"]] = c
        pools[name] = d
    return sc, pools


def split(ids):
    calib = [i for i in ids if int(hashlib.sha256(i.encode()).hexdigest(), 16) % 2 == 0]
    return calib, [i for i in ids if i not in set(calib)]


def report(name, picks, sc, ids, calib):
    """picks: id -> 선택한 답(또는 None=baseline 유지). 점수/분할을 출력한다."""
    calibs = set(calib)
    agg = {"all": [0, 0, 0, 0], "calib": [0, 0, 0, 0], "valid": [0, 0, 0, 0]}
    for i in ids:
        s = sc[i]
        pick = picks.get(i)
        final = s["base"] if pick is None else pick
        keys = ["all", "calib" if i in calibs else "valid"]
        for k in keys:
            agg[k][0] += int(final == s["ans"])
            agg[k][3] += 1
            if pick is not None and pick != s["base"]:
                if pick == s["ans"]:
                    agg[k][1] += 1
                elif s["base"] == s["ans"]:
                    agg[k][2] += 1
    base_all = sum(1 for i in ids if sc[i]["base"] == sc[i]["ans"])
    bc = sum(1 for i in ids if i in calibs and sc[i]["base"] == sc[i]["ans"])
    bv = base_all - bc
    a, c, v = agg["all"], agg["calib"], agg["valid"]
    print(f"  {name:<34} {a[0]:>3}/{a[3]:<3} {a[0]-base_all:>+4}  "
          f"gain {a[1]:>2} reg {a[2]:<2} | calib {c[0]-bc:>+3} valid {v[0]-bv:>+3}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify", type=Path, required=True, help="verify_candidates_client 출력")
    ap.add_argument("--max-candidates", type=int, default=4)
    args = ap.parse_args()

    sc, pools = load_state()
    ver = {}
    for line in open(args.verify):
        r = json.loads(line)
        ver[r["id"]] = {norm(k): v for k, v in r["candidates"].items()}

    ids = sorted(i for i in ver if i in sc)
    calib, valid = split(ids)
    base = sum(1 for i in ids if sc[i]["base"] == sc[i]["ans"])
    print(f"대상 {len(ids)}문제 (calib {len(calib)} / valid {len(valid)}), baseline {base}/{len(ids)}\n")

    def tir_counts(i):
        tot = Counter()
        for d in pools.values():
            tot += d.get(i, Counter())
        return tot

    # 오라클 참고치
    ora = sum(1 for i in ids
              if sc[i]["ans"] == sc[i]["base"] or sc[i]["ans"] in tir_counts(i))
    orav = sum(1 for i in ids if sc[i]["ans"] in ver[i] or sc[i]["ans"] == sc[i]["base"])
    print(f"오라클: TIR 후보 전체 {ora}/{len(ids)} | 역검증에 올린 후보 {orav}/{len(ids)}\n")

    print("== 기준선 ==")
    # 현행 배포: 단일 계보 min-count 2
    single = pools.get("hybrid3145", {})
    def plur(c, mc):
        top = c.most_common()
        if top and top[0][1] >= mc and not (len(top) > 1 and top[0][1] == top[1][1]):
            return top[0][0]
        return None
    report("현행 배포 (1계보 mc2)", {i: plur(single.get(i, Counter()), 2) for i in ids}, sc, ids, calib)
    report("risky>=1 + mc1", {i: (plur(single.get(i, Counter()), 1)
                                  if sc[i]["nrisky"] >= 1 else None) for i in ids}, sc, ids, calib)

    print("\n== 역검증 규칙 ==")
    def verdicts(i, a):
        d = ver[i].get(a) or {}
        t = d.get("tally", {})
        return t.get("VERIFIED", 0), t.get("REFUTED", 0)

    # A. VERIFIED 최다 (동률은 TIR 표수로)
    for tmin in (1, 2, 3):
        picks = {}
        for i in ids:
            cands = list(ver[i])
            scored = [(verdicts(i, a)[0], tir_counts(i).get(a, 0), a) for a in cands]
            scored.sort(reverse=True)
            if scored and scored[0][0] >= tmin and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                picks[i] = scored[0][2]
        report(f"A. VERIFIED 최다 (>= {tmin})", picks, sc, ids, calib)

    # B. VERIFIED - REFUTED 최다
    for tmin in (1, 2):
        picks = {}
        for i in ids:
            scored = []
            for a in ver[i]:
                v, rf = verdicts(i, a)
                scored.append((v - rf, v, tir_counts(i).get(a, 0), a))
            scored.sort(reverse=True)
            if scored and scored[0][0] >= tmin and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                picks[i] = scored[0][3]
        report(f"B. VERIFIED-REFUTED 최다 (>= {tmin})", picks, sc, ids, calib)

    # C. 유일 생존자: 한 후보만 VERIFIED>0, 나머지 전부 0
    for tmin in (1, 2):
        picks = {}
        for i in ids:
            alive = [a for a in ver[i] if verdicts(i, a)[0] >= tmin]
            if len(alive) == 1:
                picks[i] = alive[0]
        report(f"C. 유일 생존자 (VERIFIED >= {tmin})", picks, sc, ids, calib)

    # D. risky 게이트 위에 역검증을 얹는다 (형태 깨끗한 문제는 손대지 않음)
    for tmin in (1, 2):
        picks = {}
        for i in ids:
            if sc[i]["nrisky"] < 1:
                continue
            scored = [(verdicts(i, a)[0], tir_counts(i).get(a, 0), a) for a in ver[i]]
            scored.sort(reverse=True)
            if scored and scored[0][0] >= tmin and (len(scored) == 1 or scored[0][0] > scored[1][0]):
                picks[i] = scored[0][2]
            else:
                p = plur(single.get(i, Counter()), 1)
                if p is not None:
                    picks[i] = p
        report(f"D. risky + 역검증(>= {tmin}), 폴백 mc1", picks, sc, ids, calib)


if __name__ == "__main__":
    main()
