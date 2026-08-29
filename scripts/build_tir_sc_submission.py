"""TIR self-consistency 결과로 831 제출본을 만든다 (챔피언 위에 override).

규칙 (holdout464 표수<=3 서브셋 87문제에서 검증):
    if (hybrid3145 N=8 최다득표 <= 3)                    # 난이도 게이트
       and (코드검증된 답들의 unique plurality >= min_count):   # SC + 실행 게이트
        답 = 그 plurality 답
    else:
        답 = 챔피언 답

검증 결과 (baseline 17/87 기준):
    min-count 1 -> 30/87 (+13) gain 14/reg 1, 정밀도 44%, 채택 41
    min-count 2 -> 29/87 (+12) gain 13/reg 1, 정밀도 65%, 채택 26   <- 기본값
    min-count 3 -> 26/87 (+9)  gain  9/reg 0, 정밀도 85%, 채택 13

min-count 2를 기본으로 쓰는 이유: 홀드아웃은 hybrid_3145(해당 구간 정답률 20%) 기준인데
실제 제출은 챔피언(5-voter, 같은 구간에서 더 높음) 위에 얹는다. 정밀도가 낮으면
챔피언이 맞힌 답을 덮어쓸 위험이 커지므로 정밀도를 우선한다.

⚠️ 추론 시점 코드 실행은 대회 규정 회색지대다. CONTEXT "TIR" 절 참고. 백업 제출본 유지 필수.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CHAMPION = ROOT / "submission_self_consistency_hybrid3145_n8_min4_support4.csv"
TEST = ROOT / "data/deep_chal_math_leaderboard_filtered.csv"
SC_FILES = [
    "outputs/self_consistency_hybrid3145_n8_leaderboard_support1to3.jsonl",
    "outputs/self_consistency_hybrid3145_n8_leaderboard_support4.jsonl",
    "outputs/self_consistency_hybrid3145_n8_leaderboard_support5.jsonl",
]


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
    ap = argparse.ArgumentParser()
    ap.add_argument("--tir", type=Path, default=ROOT / "outputs/tir_sc8_831_vote3_to60.jsonl")
    ap.add_argument("--vote-threshold", type=int, default=3)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = list(csv.DictReader(open(CHAMPION, encoding="utf-8-sig")))
    key = "prediction" if "prediction" in rows[0] else list(rows[0].keys())[1]
    champion = {r["id"]: norm(r[key]) for r in rows}
    test_ids = [r["id"] for r in csv.DictReader(open(TEST, encoding="utf-8-sig"))]

    pool = {}
    for f in SC_FILES:
        for line in open(ROOT / f):
            r = json.loads(line)
            pool[r["id"]] = [x for x in (norm(y) for y in r.get("sample_predictions", [])) if x is not None]

    tir = {}
    for line in open(args.tir):
        r = json.loads(line)
        tir[r["id"]] = r

    def votes(i):
        c = Counter(pool.get(i, []))
        return c.most_common(1)[0][1] if c else 0

    needed = [i for i in test_ids if votes(i) <= args.vote_threshold]
    missing = [i for i in needed if i not in tir]
    if missing:
        raise SystemExit(f"게이트 대상 {len(needed)}개 중 TIR 결과 누락 {len(missing)}개")
    print(f"난이도 게이트(표수<={args.vote_threshold}) 대상 {len(needed)}문제, TIR 결과 확보")

    final, adopted, changed = {}, [], []
    for i in test_ids:
        pick = None
        if votes(i) <= args.vote_threshold and i in tir:
            counts = Counter({norm(k): v for k, v in tir[i]["verified_counts"].items() if norm(k) is not None})
            top = counts.most_common()
            if top and top[0][1] >= args.min_count and not (len(top) > 1 and top[0][1] == top[1][1]):
                pick = top[0][0]
        if pick is None:
            final[i] = champion[i]
        else:
            final[i] = pick
            adopted.append(i)
            if pick != champion[i]:
                changed.append(i)

    bad = [i for i in test_ids if final[i] is None]
    if bad:
        raise SystemExit(f"정수가 아닌 답 {len(bad)}개")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.writer(f)
        w.writerow(["id", key])
        for i in test_ids:
            w.writerow([i, final[i]])

    print(f"규칙: 표수<={args.vote_threshold} + SC 코드검증 plurality >= {args.min_count}")
    print(f"  채택      {len(adopted)}문제")
    print(f"  실제 변경 {len(changed)}문제 (나머지는 TIR이 챔피언과 같은 답)")
    print(f"  출력      {args.output} ({len(test_ids)}행)")

    # 무결성 검증
    out_rows = list(csv.DictReader(open(args.output, encoding="utf-8-sig")))
    assert [r["id"] for r in out_rows] == test_ids, "id 순서 불일치"
    assert all(r[key].lstrip("-").isdigit() for r in out_rows), "정수 아닌 값 존재"
    print(f"  검증      {len(out_rows)}행, id 순서 일치, 정수 아님 0, 중복 0")


if __name__ == "__main__":
    main()
