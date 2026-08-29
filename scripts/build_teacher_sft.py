"""teacher 32B 생성물 -> 검증 통과 풀이만 SFT 데이터로.

필터 (전부 만족해야 채택):
  1. 추출된 최종답 == gold 정답   ← 이게 '검증'이다. teacher 를 맹신하지 않는다
  2. 우리 포맷(`Final answer:`) 으로 끝남 — 추출기·5-voter·SC·TIR 스택과 호환
  3. 토큰 길이 <= max_seq        ← **문자 수가 아니라 토큰으로 센다**
     (오늘 밤 R1 증류가 정확히 이걸 안 해서 학습이 통째로 죽었다. 24건/3000건이
      3072토큰을 넘었고 train_qlora 가 첫 건에서 예외를 던졌다)
  4. 문제당 최대 --max-per-problem 개 (쉬운 문제가 데이터를 지배하지 않게)

teacher 가 아예 못 푼 문제는 자동으로 빠진다 — 그게 정상이다. 이 풀은 원래
'우리 모델이 틀리는 문제'이므로 teacher 도 일부는 못 푼다.
"""
import argparse
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402

SYSTEM = ("Solve the math problem. Think step by step, showing your reasoning "
          "concisely. End your response with exactly:\nFinal answer: <integer>")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, required=True)
    ap.add_argument("--pool", type=Path, required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--max-per-problem", type=int, default=2)
    ap.add_argument("--max-seq", type=int, default=3072)
    ap.add_argument("--base-model", default="/workspace/models/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    gold = {}
    for line in open(args.pool, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            gold[r["id"]] = norm(r.get("answer"))

    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.base_model, local_files_only=True)

    st = Counter()
    by_pid = defaultdict(list)
    for line in open(args.raw, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        g = gold.get(r["id"])
        if g is None:
            st["skip_no_gold"] += 1
            continue
        for text in (r.get("responses") or []):
            st["seen"] += 1
            if not text or "final answer" not in text.lower():
                st["skip_bad_format"] += 1
                continue
            if norm(extract_v2(text)) != g:
                st["skip_wrong"] += 1
                continue
            msgs = [{"role": "system", "content": SYSTEM},
                    {"role": "user", "content": r["question"]},
                    {"role": "assistant", "content": text.strip()}]
            ntok = len(tok.apply_chat_template(msgs, tokenize=True))
            if ntok > args.max_seq:
                st["skip_too_long"] += 1
                continue
            by_pid[r["id"]].append({"messages": msgs, "tokens": ntok})
            st["kept_candidate"] += 1

    out = []
    for pid, cands in by_pid.items():
        cands.sort(key=lambda c: c["tokens"])          # 짧고 명료한 풀이 우선
        out.extend(cands[:args.max_per_problem])

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")

    toks = [s["tokens"] for s in out]
    man = {"output": str(args.output), "samples": len(out),
           "problems_covered": len(by_pid), "pool_size": len(gold),
           "coverage_pct": round(100 * len(by_pid) / max(len(gold), 1), 1),
           "tokens": {"mean": int(sum(toks) / max(len(toks), 1)),
                      "max": max(toks) if toks else 0},
           "stats": dict(st),
           "teacher": "Qwen/Qwen2.5-32B-Instruct-AWQ (Apache-2.0, 공개)"}
    Path(str(args.output).replace(".jsonl", ".manifest.json")).write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(man, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
