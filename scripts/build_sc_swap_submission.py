"""SC 담당 교체 제출본 빌더 — 현행 660(pool24 v3mc2) 위에서 support4 SC override 의
샘플러만 GRPO 체크포인트 N=8 로 바꾼다. 홀드아웃 검증(13절: 382→387, calib/valid 양수)과
동일 규칙: 교체는 TIR 미개입 문제(현행 답 == 챔피언623 답)에만 반영.

사용:
/usr/bin/python3 scripts/build_sc_swap_submission.py \
  --ck-n8 outputs/ck150_n8_leaderboard_support4.jsonl \
  --output submissions/submission_sc_swap_ck150.csv
"""
import argparse
import csv
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402

SUPPORT4 = "outputs/self_consistency_hybrid3145_n8_leaderboard_support4.jsonl"
BASE_SUB = "submissions/submission_pool24_v3mc2.csv"


def jl(rel):
    return [json.loads(l) for l in open(ROOT / rel) if l.strip()]


def override(preds):
    c = Counter(p for p in preds if p is not None)
    tp = c.most_common()
    if tp and tp[0][1] >= 4 and (len(tp) == 1 or tp[0][1] > tp[1][1]):
        return tp[0][0]
    return None


ap = argparse.ArgumentParser()
ap.add_argument("--ck-n8", required=True)
ap.add_argument("--output", required=True)
args = ap.parse_args()

cur = {}
with open(ROOT / BASE_SUB, encoding="utf-8-sig", newline="") as f:
    rows = list(csv.DictReader(f))
    cols = rows[0].keys()
    for r in rows:
        cur[r["id"]] = r["answer"]

ck = {r["id"]: [norm(x) for x in r["predictions"]] for r in jl(args.ck_n8)}

changes = []
for r in jl(SUPPORT4):
    i = r["id"]
    base5 = norm(r["baseline_prediction"])  # 5-voter (override 이전)
    hyb = [norm(extract_v2(t)) for t in (r.get("responses") or [])] or \
          [norm(x) for x in (r.get("sample_predictions") or [])]
    base623 = override(hyb)
    base623 = base623 if base623 is not None else base5
    swapped = override(ck.get(i, []))
    swapped = swapped if swapped is not None else base5
    cur_ans = norm(cur[i])
    if swapped != base623 and cur_ans == base623:
        changes.append((i, str(base623), str(swapped)))
        cur[i] = str(swapped)

out_rows = [{k: (cur[r["id"]] if k == "answer" else r[k]) for k in cols} for r in rows]
with open(ROOT / args.output, "w", newline="", encoding="utf-8") as f:
    w = csv.DictWriter(f, fieldnames=list(cols))
    w.writeheader()
    w.writerows(out_rows)

# 무결성
assert len(out_rows) == 831, len(out_rows)
assert len({r["id"] for r in out_rows}) == 831
bad = [r for r in out_rows if not str(r["answer"]).lstrip("-").isdigit()]
assert not bad, bad[:3]
print(f"현행 660 대비 변경: {len(changes)}개")
for i, a, b in changes:
    print(f"  {i}: {a} -> {b}")
print(f"831행/ID중복0/전부정수 검증 통과 -> {args.output}")
