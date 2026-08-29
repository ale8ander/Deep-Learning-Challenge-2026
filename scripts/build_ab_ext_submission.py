"""A+B 확장 제출본 빌더 — 665 챔피언(삼중 게이트) 위에 두 확장을 얹는다.

  A (게이트 완화): ck150 N=8 유일최빈이 정확히 4표(5표 이상은 이미 665 에 반영)이고
     현행 답과 다르면 교체 후보 — 코드가드(기존 답의 코드 표가 더 많으면 취소) 유지.
  B (순수 코드 override): A 미적용 문제 중, TIR 4풀 합산 verified_counts 의 유일최빈이
     2표 이상이고 현행 답과 다르며 **현행 답의 코드 표가 0**이면 코드 답 채택.

홀드아웃 실측(665 등가 387 기준): 합본 +4 (변경 12, gain 6/reg 2, calib +3/valid +1).

사용:
  /usr/bin/python3 scripts/build_ab_ext_submission.py \
    --output submissions/submission_ab_gate4_codeoverride.csv [--reference <csv> --verify-only]
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import norm  # noqa: E402
from tir_common import normalize as tnorm  # noqa: E402

BASE_SUB = "submissions/submission_ck150_gate5_sup4_codeguard.csv"   # 665 챔피언
CK_N8 = ["outputs/ck150_n8_leaderboard_support4.jsonl",
         "outputs/ck150_n8_leaderboard_support1to3.jsonl"]
TIR_POOLS = ["outputs/tir_sc8_831_vote3_to60.jsonl",
             "outputs/tir_repair1_831_gate282.jsonl",
             "outputs/tir_nocode_831_gate282.jsonl",
             "outputs/tirc_831_vote45.jsonl"]


def jl(rel):
    return [json.loads(l) for l in open(ROOT / rel) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--reference", default=None)
    ap.add_argument("--verify-only", action="store_true")
    args = ap.parse_args()

    with open(ROOT / BASE_SUB, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        cols = list(rows[0].keys())
    cur = {r["id"]: r["answer"] for r in rows}

    ver = {}
    for rel in TIR_POOLS:
        for r in jl(rel):
            c = ver.setdefault(r["id"], Counter())
            for a, v in (r.get("verified_counts") or {}).items():
                a = tnorm(a)
                if a is not None:
                    c[a] += v

    a_flips, a_guard = [], []
    for rel in CK_N8:
        for r in jl(rel):
            i = r["id"]
            c = Counter(norm(x) for x in r["predictions"] if norm(x) is not None)
            tp = c.most_common()
            if not tp or (len(tp) > 1 and tp[0][1] == tp[1][1]) or tp[0][1] != 4:
                continue
            mode = str(tp[0][0])
            if mode == str(norm(cur[i])):
                continue
            vc = ver.get(i, Counter())
            if vc and vc.get(tnorm(cur[i]), 0) > vc.get(tnorm(mode), 0):
                a_guard.append(i)
                continue
            a_flips.append((i, cur[i], mode))
            cur[i] = mode

    a_ids = {i for i, _, _ in a_flips}
    b_flips = []
    for i, vc in ver.items():
        if i in a_ids or i not in cur or not vc:
            continue
        tp = vc.most_common()
        if tp and tp[0][1] >= 2 and (len(tp) == 1 or tp[0][1] > tp[1][1]):
            mode = str(tp[0][0])
            if mode != str(norm(cur[i])) and vc.get(tnorm(cur[i]), 0) == 0:
                b_flips.append((i, cur[i], mode))
                cur[i] = mode

    out = [{k: (cur[r["id"]] if k == "answer" else r[k]) for k in cols} for r in rows]
    assert len(out) == 831 and len({r["id"] for r in out}) == 831
    assert all(str(r["answer"]).lstrip("-").isdigit() for r in out)
    print(f"A(ck 4표+가드): {len(a_flips)}개 교체 / 가드취소 {len(a_guard)}개")
    for i, o, n_ in a_flips:
        print(f"  A {i}: {o} -> {n_}")
    print(f"B(코드 2표+ & 현행 0표): {len(b_flips)}개 교체")
    for i, o, n_ in b_flips:
        print(f"  B {i}: {o} -> {n_}")

    if args.reference:
        ref = {r["id"]: r["answer"] for r in csv.DictReader(
            open(ROOT / args.reference, encoding="utf-8-sig"))}
        mism = [r["id"] for r in out if str(ref[r["id"]]) != str(r["answer"])]
        if mism:
            print(f"[불일치] {len(mism)}개: {mism[:5]}")
            sys.exit(1)
        print(f"[통과] {args.reference} 와 831문제 전부 일치")
    if not args.verify_only:
        with open(ROOT / args.output, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=cols)
            w.writeheader()
            w.writerows(out)
        print(f"831행/중복0/정수 검증 통과 -> {args.output}")


if __name__ == "__main__":
    main()
