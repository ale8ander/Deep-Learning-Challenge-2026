"""최종샷 빌더 — 게이트 v3 (ck150 N=64 집중도 >=0.425 & support<=4 & 코드가드).

홀드아웃 검증: 665 등가 387 -> 392 (+5, gain 5/reg 0). 베이스는 665 챔피언 CSV.
결정론적 — reproduce_all compose 재현 검증 대상.

사용:
  /usr/bin/python3 scripts/build_final_union_submission.py \
    --output submissions/submission_final_gate425.csv [--reference <csv> --verify-only]
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
CK_N64 = ["outputs/ck150_n64_lb_support4.jsonl",
          "outputs/ck150_n64_lb_support1to3.jsonl"]
TIR_POOLS = ["outputs/tir_sc8_831_vote3_to60.jsonl",
             "outputs/tir_repair1_831_gate282.jsonl",
             "outputs/tir_nocode_831_gate282.jsonl",
             "outputs/tirc_831_vote45.jsonl"]
FRAC = 0.425


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

    flips, guarded = [], []
    for rel in CK_N64:
        for r in jl(rel):
            i = r["id"]
            preds = [norm(x) for x in r["predictions"] if norm(x) is not None]
            if not preds:
                continue
            c = Counter(preds)
            tp = c.most_common()
            if len(tp) > 1 and tp[0][1] == tp[1][1]:
                continue
            if tp[0][1] / len(preds) < FRAC:
                continue
            m = str(tp[0][0])
            if m == str(norm(cur[i])):
                continue
            vc = ver.get(i, Counter())
            if vc and vc.get(tnorm(cur[i]), 0) > vc.get(tnorm(m), 0):
                guarded.append(i)
                continue
            flips.append((i, cur[i], m, f"{tp[0][1]}/{len(preds)}"))
            cur[i] = m

    out = [{k: (cur[r["id"]] if k == "answer" else r[k]) for k in cols} for r in rows]
    assert len(out) == 831 and len({r["id"] for r in out}) == 831
    assert all(str(r["answer"]).lstrip("-").isdigit() for r in out)
    print(f"게이트 v3 교체 {len(flips)}개 / 코드가드 취소 {len(guarded)}개")
    for i, o, n_, conc in flips:
        print(f"  {i}: {o} -> {n_} ({conc})")

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
