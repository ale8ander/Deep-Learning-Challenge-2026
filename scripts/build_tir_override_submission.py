"""현행 챔피언 CSV 위에 TIR override를 얹어 831 제출본을 만든다.

규칙 (holdout464에서 검증: 364/464, +7, gain 8 / reg 1, calib +3 / valid +4):
    if (기존 hybrid3145 N=8 최다득표 <= 3)          # 난이도 게이트
       and (코드 실행 ok and stdout 정수 and stdout == 최종답):   # 실행 게이트
        답 = TIR 답
    else:
        답 = 챔피언 답

--require-low-support 를 주면 5-voter support <= 3 게이트를 추가한다(B안).
챔피언이 만장일치이거나 이미 SC override를 적용한 구간을 건드리지 않는 보수적 버전이며,
이 추가 게이트는 holdout464에서 검증되지 않았다(voter 예측 부재).

⚠️ 추론 시점 코드 실행은 대회 규정 회색지대다. CONTEXT "TIR 스모크" 절 참고.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAMPION = ROOT / "submission_self_consistency_hybrid3145_n8_min4_support4.csv"
TEST = ROOT / "data/deep_chal_math_leaderboard_filtered.csv"
TIR_FILES = [
    "outputs/tir_831_vote3_hybrid3145.jsonl",      # 게이트 통과분 전용 실행
    "outputs/tir_leaderboard831_partial.jsonl",    # 전체 실행 중단분 재사용
    "outputs/tir_leaderboard831_hybrid3145.jsonl", # 전체 실행 완주분(있으면)
]
SC_FILES = [
    "outputs/self_consistency_hybrid3145_n8_leaderboard_support1to3.jsonl",
    "outputs/self_consistency_hybrid3145_n8_leaderboard_support4.jsonl",
    "outputs/self_consistency_hybrid3145_n8_leaderboard_support5.jsonl",
]
VOTERS = [
    "submission_hybrid_3145.csv",
    "submission_hybrid_3244.csv",
    "submission_external_3000.csv",
    "submission_hybrid_4145_r8_qv_lr1p5e6_e1_retry2048.csv",
    "submission_hybrid_3145_verify_retry2048.csv",
]


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


def load_submission(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    key = "prediction" if "prediction" in rows[0] else list(rows[0].keys())[1]
    return {r["id"]: norm(r[key]) for r in rows}, key


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--vote-threshold", type=int, default=3)
    ap.add_argument("--require-low-support", action="store_true")
    ap.add_argument("--support-threshold", type=int, default=3)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    champion, champ_key = load_submission(CHAMPION)
    test_rows = list(csv.DictReader(open(TEST, encoding="utf-8-sig")))
    test_ids = [r["id"] for r in test_rows]

    pool = {}
    for f in SC_FILES:
        for line in open(ROOT / f):
            r = json.loads(line)
            pool[r["id"]] = [x for x in (norm(y) for y in r.get("sample_predictions", [])) if x is not None]

    tir = {}
    for path in TIR_FILES:
        p = ROOT / path
        if not p.exists():
            continue
        for line in open(p):
            r = json.loads(line)
            tir.setdefault(r["id"], r)

    voters = [load_submission(ROOT / v)[0] for v in VOTERS]

    def votes(i):
        c = Counter(pool.get(i, []))
        return c.most_common(1)[0][1] if c else 0

    def support(i):
        c = Counter(v[i] for v in voters if v.get(i) is not None)
        return c.most_common(1)[0][1] if c else 0

    def exec_gate(i):
        r = tir.get(i)
        if r is None or r.get("exec_status") != "ok":
            return False
        so = norm((r.get("exec_stdout") or "").strip())
        return so is not None and so == norm(r["prediction"])

    # 난이도 게이트를 통과한 문제에만 TIR 예측이 필요하다(나머지는 챔피언 답 그대로).
    needed = [i for i in test_ids if votes(i) <= args.vote_threshold]
    missing = [i for i in needed if i not in tir]
    if missing:
        raise SystemExit(
            f"게이트 대상 {len(needed)}개 중 TIR 예측 누락 {len(missing)}개 (예: {missing[:5]})"
        )
    print(f"난이도 게이트(표수<={args.vote_threshold}) 대상 {len(needed)}문제, TIR 예측 확보 완료")

    fired, changed, final = [], [], {}
    for i in test_ids:
        ok = votes(i) <= args.vote_threshold and exec_gate(i)
        if ok and args.require_low_support:
            ok = support(i) <= args.support_threshold
        if ok:
            fired.append(i)
            answer = norm(tir[i]["prediction"])
            if answer is None:
                answer = champion[i]
            elif answer != champion[i]:
                changed.append(i)
            final[i] = answer
        else:
            final[i] = champion[i]

    bad = [i for i in test_ids if final[i] is None]
    if bad:
        raise SystemExit(f"정수가 아닌 답 {len(bad)}개")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", champ_key])
        for i in test_ids:
            w.writerow([i, final[i]])

    print(f"규칙: 표수<={args.vote_threshold} + 실행게이트"
          + (f" + support<={args.support_threshold}" if args.require_low_support else ""))
    print(f"  발동      {len(fired)}문제")
    print(f"  실제 변경 {len(changed)}문제 (나머지는 TIR이 챔피언과 같은 답)")
    print(f"  출력      {args.output}  ({len(test_ids)}행)")
    if changed:
        sup = Counter(support(i) for i in changed)
        vt = Counter(votes(i) for i in changed)
        print(f"  변경 문제의 5-voter support 분포: {dict(sorted(sup.items()))}")
        print(f"  변경 문제의 N=8 표수 분포      : {dict(sorted(vt.items()))}")


if __name__ == "__main__":
    main()
