"""few-shot 포인터 게이트 제출본 빌더 (2026-08-30, 홀드아웃 사전 등록 규칙).

홀드아웃464 실측: 교체 12, gain 9 / reg 0, 델타 +9 (calib +5 / valid +4).

규칙 (사전 등록, 스윕 재조정 금지):
  1. 포인터: few-shot(3-shot) greedy 답 a 가 현행 제출본(665) 답과 다름
  2. 확인 1: 2계보 16샘플 풀(ck150 N=8 + hybrid3145 N=8)에서 a 표 >= 4
  3. 확인 2: 상대우위 — a 표 > 현행답 표 (같은 풀)
  4. 확인 3: 자기재현 — few-shot stochastic N=8 에서 a 표 >= 2
  5. support 캡 없음(<=5 전부), 코드가드 없음 (홀드아웃에서 가드가 회수 4개를 죽였음)

재료는 전부 이 pod(5090)에서 생성돼 계보 일관성이 있다 (A100 산출물 혼용 금지 규칙 준수).
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

BASE_SUB = "submissions/submission_ck150_gate5_sup4_codeguard.csv"  # 665
FS_GREEDY = "outputs/fewshot3_h3145_831_greedy.jsonl"
FS_N8 = "outputs/fewshot3_h3145_n8_831_seed20260925.jsonl"
POOLS = ["outputs/ck150_n8lp_831_seed20260924.jsonl",
         "outputs/h3145_n8lp_831_seed20260924.jsonl"]


def jl(rel):
    return [json.loads(l) for l in open(ROOT / rel) if l.strip()]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--output", required=True)
    ap.add_argument("--min-votes", type=int, default=4)
    ap.add_argument("--min-fs8", type=int, default=2)
    args = ap.parse_args()

    with open(ROOT / BASE_SUB, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
        cols = list(rows[0].keys())
    cur = {r["id"]: str(r["answer"]) for r in rows}

    fsg = {r["id"]: (None if r["prediction"] is None else str(r["prediction"]))
           for r in jl(FS_GREEDY)}
    fs8 = {r["id"]: [p for p in r.get("predictions", [])] for r in jl(FS_N8)}
    pool = {}
    for rel in POOLS:
        for r in jl(rel):
            pool.setdefault(r["id"], []).extend(
                None if p is None else str(norm(p)) for p in r["predictions"])

    def votes(i, a):
        return sum(1 for p in pool.get(i, []) if p == a)

    flips = []
    for i in cur:
        a = fsg.get(i)
        if a is None or a == str(norm(cur[i])):
            continue
        if votes(i, a) < args.min_votes:
            continue
        if votes(i, a) <= votes(i, str(norm(cur[i]))):
            continue
        if sum(1 for p in fs8.get(i, []) if p == a) < args.min_fs8:
            continue
        flips.append((i, a))
        cur[i] = a

    out = [{k: (cur[r["id"]] if k == "answer" else r[k]) for k in cols} for r in rows]
    assert len(out) == 831 and len({r["id"] for r in out}) == 831
    assert all(str(r["answer"]).lstrip("-").isdigit() for r in out)
    print(f"교체 {len(flips)}개:")
    for i, a in flips:
        print(f"  {i}: -> {a}")
    with open(ROOT / args.output, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=cols)
        w.writeheader()
        w.writerows(out)
    print(f"저장: {args.output}")


if __name__ == "__main__":
    main()
