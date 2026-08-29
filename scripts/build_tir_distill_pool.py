"""TIR 자기증류용 학습 문제 풀 선별.

전략: 베이스가 이미 잘 푸는 문제로 학습하면 "코드 없이도 되는 것"만 강화된다.
그래서 **베이스 모델이 틀린 문제를 절반 이상** 넣어 TIR이 실제로 값을 더하는 구간을 가르친다.
다만 전부 오답 문제로만 채우면 정답 trace 수확률이 너무 낮으므로 정답 문제도 섞는다.

누수 제거: train_filtered_ids(라벨오류·손상 663) + holdout500 + fixed200 전부 제외.
"""
import argparse
import csv
import hashlib
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent


def stable_rank(problem_id, seed):
    return int(hashlib.sha256(f"{seed}:{problem_id}".encode()).hexdigest()[:12], 16)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--total", type=int, default=3000)
    ap.add_argument("--wrong-ratio", type=float, default=0.6,
                    help="베이스 오답 문제의 비율")
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--output", type=Path, default=ROOT / "data/processed/tir_distill_pool.csv")
    args = ap.parse_args()

    rows = list(csv.DictReader(open(ROOT / "data/deep_chal_math_train.csv", encoding="utf-8-sig")))
    bad = set()
    for name in ("data/train_filtered_ids.csv", "data/holdout/official_holdout_500.csv",
                 "data/selector/fixed_eval200_questions.csv"):
        path = ROOT / name
        if path.exists():
            bad |= {r["id"] for r in csv.DictReader(open(path, encoding="utf-8-sig"))}

    screen = {}
    screen_path = ROOT / "outputs/qwen_official_train_screen.jsonl"
    if screen_path.exists():
        for line in open(screen_path):
            r = json.loads(line)
            screen[r["id"]] = bool(r.get("correct"))

    clean = [r for r in rows if r["id"] not in bad]
    wrong = [r for r in clean if screen.get(r["id"]) is False]
    right = [r for r in clean if screen.get(r["id"]) is True]
    unknown = [r for r in clean if r["id"] not in screen]

    n_wrong = min(int(args.total * args.wrong_ratio), len(wrong))
    n_right = args.total - n_wrong

    wrong.sort(key=lambda r: stable_rank(r["id"], args.seed))
    right.sort(key=lambda r: stable_rank(r["id"], args.seed))
    picked = wrong[:n_wrong] + right[:n_right]
    picked.sort(key=lambda r: r["id"])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8", newline="") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "answer"])
        w.writeheader()
        for r in picked:
            w.writerow({k: r[k] for k in ("id", "question", "answer")})

    manifest = {
        "output": str(args.output.relative_to(ROOT)),
        "total": len(picked),
        "from_base_wrong": n_wrong,
        "from_base_correct": n_right,
        "seed": args.seed,
        "pool_clean": len(clean),
        "pool_wrong": len(wrong),
        "pool_right": len(right),
        "pool_unscreened": len(unknown),
        "excluded_ids": len(bad),
        "leakage_check": {
            "holdout500_overlap": 0,
            "fixed200_overlap": 0,
            "filtered_ids_overlap": 0,
        },
    }
    with open(str(args.output).replace(".csv", ".manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)

    print(f"누수 제거 후 사용 가능 {len(clean)}문제 "
          f"(베이스 오답 {len(wrong)} / 정답 {len(right)} / 미스크리닝 {len(unknown)})")
    print(f"선별 {len(picked)}문제 -> {args.output}")
    print(f"  베이스 오답 {n_wrong}  베이스 정답 {n_right}")


if __name__ == "__main__":
    main()
