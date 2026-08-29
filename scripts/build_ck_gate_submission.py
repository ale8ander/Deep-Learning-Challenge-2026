"""ck 게이트 + 코드가드 제출본 빌더 (결정론적, 재현 검증용).

규칙 (2026-08-29 오전 세션 13~16절 + 코드가드):
  1. 대상: 5-voter support<=4 문제 (support1to3 + support4 파일의 합집합, 297문제)
  2. 게이트: ck150 N=8 의 유일 최빈값이 5표 이상이고 현행(660) 답과 다르면 교체 후보
  3. 코드가드: TIR 4풀 합산 verified_counts 에서 기존 답 표 > 새 답 표이면 교체 취소

사용:
  /usr/bin/python3 scripts/build_ck_gate_submission.py \
    --output submission_ck150_gate5_sup4_codeguard.csv [--reference <csv> --verify-only]
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

BASE_SUB = "submission_pool24_v3mc2.csv"
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

    kept, dropped = [], []
    for rel in CK_N8:
        for r in jl(rel):
            i = r["id"]
            c = Counter(norm(x) for x in r["predictions"] if norm(x) is not None)
            tp = c.most_common()
            if not tp or (len(tp) > 1 and tp[0][1] == tp[1][1]) or tp[0][1] < 5:
                continue
            mode = str(tp[0][0])
            if mode == str(norm(cur[i])):
                continue
            vc = ver.get(i, Counter())
            vo, vg = vc.get(tnorm(cur[i]), 0), vc.get(tnorm(mode), 0)
            if vc and vo > vg:
                dropped.append(i)
            else:
                kept.append(i)
                cur[i] = mode

    out = [{k: (cur[r["id"]] if k == "answer" else r[k]) for k in cols} for r in rows]
    assert len(out) == 831 and len({r["id"] for r in out}) == 831
    assert all(str(r["answer"]).lstrip("-").isdigit() for r in out)
    print(f"게이트 교체 {len(kept)}개 / 코드가드 취소 {len(dropped)}개")

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
        print(f"저장: {args.output}")


if __name__ == "__main__":
    main()
