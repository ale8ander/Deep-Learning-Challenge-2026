"""TIR 궤적 덤프 -> outcome-verifier 4차 시도 학습 데이터.

── 왜 4차인가 (3연패 전력과 다른 점) ────────────────────────────────
과거: 이진분류 dev 59% / pairwise 62.5~68% / outcome A/B 56% — 전부 dev 에서 기각.
이번이 다른 점 두 가지:
  1. 데이터 2.6배 — 후보 4,800 -> 12,600 (수확 12,000 + 스모크 600)
  2. 후보가 TIR 궤적이라 **코드·실행출력**이 입력에 포함된다. 산문 CoT 보다
     정오 판별의 표면 단서가 훨씬 많다.

── 사전 등록 kill switch ─────────────────────────────────────────
dev 에서 두 baseline 을 모두 이겨야 한다:
  (a) 다수 클래스 예측
  (b) **코드검증 휴리스틱** — exec ok AND stdout==답 이면 정답이라 예측.
      배포 중인 min-count 규칙이 이미 이 신호를 쓰므로, 이걸 못 이기면
      verifier 는 기존 규칙에 아무것도 더하지 못한다.
(b) 대비 +5%p 미만이면 즉시 폐기. 3연패에 4연패를 더하는 데 15분이면 충분하다.

문제 단위 split (후보 단위로 섞으면 같은 문제가 양쪽에 들어가 누수다).
"""
import argparse
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tir_common import normalize  # noqa: E402

VERIFIER_SYSTEM = (
    "You are a rigorous math solution verifier. You will see a competition math "
    "problem, a candidate solution that uses a Python program, the program's "
    "output, and the proposed final answer. Judge whether the proposed answer "
    "is correct. Reply with exactly one letter: A if correct, B if incorrect."
)


def trim(text, head=1600, tail=1200):
    """긴 궤적은 가운데를 접는다 — 도입 추론과 코드/결말이 판별에 제일 중요하다."""
    if text is None:
        return ""
    if len(text) <= head + tail + 20:
        return text
    return text[:head] + "\n...[trimmed]...\n" + text[-tail:]


def stable_split(pid, seed, dev_frac):
    h = int(hashlib.sha256(f"{seed}:{pid}".encode()).hexdigest()[:12], 16)
    return "dev" if (h % 10000) < dev_frac * 10000 else "train"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=Path, nargs="+", required=True)
    ap.add_argument("--questions", type=Path,
                    default=ROOT / "data/processed/tir_distill_pool.csv")
    ap.add_argument("--out-train", type=Path, required=True)
    ap.add_argument("--out-dev", type=Path, required=True)
    ap.add_argument("--dev-frac", type=float, default=0.12)
    ap.add_argument("--seed", type=int, default=20260903)
    args = ap.parse_args()

    import csv
    questions = {r["id"]: r["question"]
                 for r in csv.DictReader(open(args.questions, encoding="utf-8-sig"))}

    stats = Counter()
    rows = {"train": [], "dev": []}
    heur = Counter()   # 코드검증 휴리스틱 baseline 집계 (dev 만)
    for path in args.traj:
        for line in open(path, encoding="utf-8"):
            r = json.loads(line)
            stats["seen"] += 1
            pred = normalize(r.get("prediction"))
            gold = normalize(r.get("answer"))
            if pred is None or gold is None or r["id"] not in questions:
                stats["skip_no_pred_or_gold"] += 1
                continue
            correct = pred == gold
            so = normalize((r.get("exec_stdout") or "").strip())
            code_verified = r.get("exec_status") == "ok" and so is not None and so == pred
            out_body = (r.get("exec_stdout") or "").strip() or "(no output)"
            if r.get("exec_status") not in (None, "ok"):
                out_body += f"\n[{r['exec_status']}]"
            user = (
                f"Problem:\n{questions[r['id']]}\n\n"
                f"Candidate solution:\n{trim(r.get('latest_text'))}\n\n"
                f"Program output:\n{trim(out_body, 300, 100)}\n\n"
                f"Proposed final answer: {pred}\n\n"
                "Is the proposed answer correct? Reply with exactly one letter: "
                "A (correct) or B (incorrect)."
            )
            part = stable_split(r["id"], args.seed, args.dev_frac)
            rows[part].append({
                "messages": [
                    {"role": "system", "content": VERIFIER_SYSTEM},
                    {"role": "user", "content": user},
                    {"role": "assistant", "content": "A" if correct else "B"},
                ],
                "meta": {"id": r["id"], "correct": correct,
                         "code_verified": code_verified,
                         "prediction": str(pred)},
            })
            stats[f"{part}_{'pos' if correct else 'neg'}"] += 1
            if part == "dev":
                heur["total"] += 1
                heur["hit"] += int(code_verified == correct)

    rng = random.Random(args.seed)
    for part, out in ((("train"), args.out_train), (("dev"), args.out_dev)):
        rng.shuffle(rows[part])
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            for s in rows[part]:
                f.write(json.dumps(s, ensure_ascii=False) + "\n")

    n_tr = len(rows["train"]); n_dev = len(rows["dev"])
    dev_pos = stats["dev_pos"]
    manifest = {
        "train": n_tr, "dev": n_dev,
        "train_pos_ratio": round(stats["train_pos"] / max(n_tr, 1), 4),
        "dev_pos_ratio": round(dev_pos / max(n_dev, 1), 4),
        "baseline_majority_dev": round(max(dev_pos, n_dev - dev_pos) / max(n_dev, 1), 4),
        "baseline_code_verified_dev": round(heur["hit"] / max(heur["total"], 1), 4),
        "stats": dict(stats), "seed": args.seed,
        "kill_switch": "dev acc 가 baseline_code_verified_dev + 0.05 미만이면 폐기",
    }
    Path(str(args.out_train).replace(".jsonl", ".manifest.json")).write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
