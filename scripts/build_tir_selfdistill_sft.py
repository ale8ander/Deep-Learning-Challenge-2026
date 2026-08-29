"""TIR 리페어 하네스의 궤적 덤프 -> 자기증류 SFT 데이터.

── 왜 이 실험은 기각된 'TIR SFT 재시도 금지'(22절)에 안 걸리는가 ─────────────
22절의 두 실패는 (a) fresh base 에 (b) 외부 teacher(NuminaMath) 코드를 (c) TIR-only 로
학습한 것이다. 이 스크립트는 세 변수를 전부 뒤집는다:
  (a) hybrid_3145 어댑터에서 **이어서**(continuation) 학습하고,
  (b) **이 모델 자신이 검증 통과한** 코드만 쓰며 (스타일 격차 원천 차단),
  (c) hybrid_3145 원본 데이터를 **replay 로 섞어** 일반 능력 붕괴를 막는다.
CONTEXT 요약 23행이 이 조합을 "미검증이므로 일반 SFT와 분리해 다시 열어둔다"고 명시했다.

── 역사 재작성 (핵심 트릭) ─────────────────────────────────────────
리페어/코드 미생성 재시도로 살아난 샘플의 최종 코드를 **1라운드 응답인 것처럼** 학습한다.
이유: 17절의 자기증류 반론("이미 성공한 코드만 강화")은 organic 성공 샘플에만 해당한다.
리페어로 살아난 코드는 베이스가 첫 시도에 **못 내놓는** 행동이므로, 이걸 1라운드에
이식하는 것은 결손 행동 교정이지 기존 행동 강화가 아니다.
(게이트 실측: 코드 미생성 21.8% + 실행 에러 12% = 샘플의 34% 가 이 결손이다.)

── 필터 (전부 만족해야 채택) ────────────────────────────────────────
  1. 코드 실행 ok
  2. stdout 정수 == 최종답 == 공식 정답  (검증된 정답 궤적만)
  3. 최종답 텍스트가 실제로 그 정수를 추출시킴 (extract_answer 재확인)
  4. chat template 적용 후 max_seq 이하 (train_qlora 는 초과 시 하드 실패한다)

프롬프트는 tir_common 에서 직접 import 한다 (복붙 금지 — 8/27 불일치 결함 재발 방지).
"""
import argparse
import csv
import hashlib
import json
import random
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tir_common import TIR_SYSTEM, FINAL_NUDGE, extract_answer, normalize  # noqa: E402


def build_messages(question, code_text, stdout, final_text):
    """추론 시점과 **동일한** 5턴 포맷 (tir_repair_client 의 fb 문자열과 일치)."""
    body = (stdout or "").strip() or "(no output)"
    return [
        {"role": "system", "content": TIR_SYSTEM},
        {"role": "user", "content": question},
        {"role": "assistant", "content": code_text},
        {"role": "user", "content": f"Program output:\n```\n{body}\n```\n{FINAL_NUDGE}"},
        {"role": "assistant", "content": final_text},
    ]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=Path, nargs="+", required=True,
                    help="tir_repair_client --dump-trajectories 산출물(들)")
    ap.add_argument("--pool", type=Path, required=True,
                    help="문제 본문 조인용 CSV (id,question,answer)")
    ap.add_argument("--replay", type=Path, default=ROOT / "data/processed/hybrid_3145.jsonl")
    ap.add_argument("--replay-ratio", type=float, default=1.0,
                    help="TIR 샘플 수 대비 replay 샘플 비율 (1.0 = 1:1)")
    ap.add_argument("--max-per-problem", type=int, default=2,
                    help="같은 문제의 궤적 상한 (쉬운 문제 과대표집 방지)")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--seed", type=int, default=20260901)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(
        "/workspace/models/Qwen2.5-3B-Instruct", local_files_only=True)

    questions = {r["id"]: r["question"]
                 for r in csv.DictReader(open(args.pool, encoding="utf-8-sig"))}

    stats = Counter()
    per_problem = {}          # id -> [(우선순위, rec)]
    for path in args.traj:
        for line in open(path, encoding="utf-8"):
            rec = json.loads(line)
            stats["seen"] += 1
            if rec["exec_status"] != "ok" or rec["prediction"] is None:
                stats["skip_not_ok"] += 1
                continue
            pred = normalize(rec["prediction"])
            gold = normalize(rec["answer"])
            so = normalize((rec["exec_stdout"] or "").strip())
            if gold is None or pred != gold or so != pred:
                stats["skip_not_verified_correct"] += 1
                continue
            if normalize(extract_answer(rec["final_text"])) != pred:
                stats["skip_final_text_mismatch"] += 1
                continue
            if rec["id"] not in questions:
                stats["skip_no_question"] += 1
                continue
            # 살아난 궤적(결손 교정)을 organic 보다 우선 채택한다.
            rescued = rec["from_nocode_retry"] or rec["from_repair"]
            per_problem.setdefault(rec["id"], []).append((0 if rescued else 1, rec))

    rng = random.Random(args.seed)
    tir_samples = []
    for pid, cands in sorted(per_problem.items()):
        rng.shuffle(cands)
        cands.sort(key=lambda t: t[0])          # rescued 먼저
        picked = cands[: args.max_per_problem]
        for prio, rec in picked:
            msgs = build_messages(questions[pid], rec["latest_text"],
                                  rec["exec_stdout"], rec["final_text"])
            n_tok = len(tokenizer.apply_chat_template(msgs, tokenize=True))
            if n_tok > args.max_seq_length:
                stats["skip_too_long"] += 1
                continue
            stats["kept_rescued" if prio == 0 else "kept_organic"] += 1
            tir_samples.append({"messages": msgs, "meta": {
                "id": pid, "rescued": prio == 0, "tokens": n_tok}})

    replay_rows = [json.loads(l) for l in open(args.replay, encoding="utf-8") if l.strip()]
    n_replay = min(len(replay_rows), int(round(len(tir_samples) * args.replay_ratio)))
    rng.shuffle(replay_rows)
    replay_take = replay_rows[:n_replay]

    combined = tir_samples + [{"messages": r["messages"], "meta": {"replay": True}}
                              for r in replay_take]
    rng.shuffle(combined)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for s in combined:
            f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")

    manifest = {
        "output": str(args.output),
        "traj_inputs": [str(p) for p in args.traj],
        "pool": str(args.pool),
        "replay": str(args.replay),
        "tir_samples": len(tir_samples),
        "tir_problems": len(per_problem),
        "replay_samples": n_replay,
        "total": len(combined),
        "seed": args.seed,
        "stats": dict(stats),
        "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest(),
    }
    mpath = Path(str(args.output).replace(".jsonl", ".manifest.json"))
    mpath.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(manifest, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
