"""추론용 system 프롬프트로 통일한 대형 SFT 코퍼스를 만든다.

발견된 버그: 기존 SFT 데이터는 전부 생성용 system 프롬프트
  "Solve the math problem independently. Give a concise, logically complete derivation.
   Do not restate the problem or explore multiple approaches. ..."
로 학습됐는데, 추론(baseline.py SYSTEM_PROMPTS["default"])은
  "Solve the math problem carefully. The answer is always an integer. ..."
를 쓴다. 즉 학습 조건과 추론 조건이 다르고, 학습 프롬프트는 명시적으로 "간결하게"를
지시하고 있었다. 이것이 (a) 약한 학습에서 데이터가 뭐든 출력이 안 변한 것과
(b) 강한 학습에서 응답이 짧아지며 붕괴한 것을 동시에 설명한다.

여기서는 system 프롬프트만 추론용으로 교체하고 문제/풀이는 그대로 둔다.
"""
import argparse
import hashlib
import json
import re
from pathlib import Path

from transformers import AutoTokenizer

ROOT = Path(__file__).resolve().parent.parent
INFERENCE_SYSTEM = (
    "Solve the math problem carefully. The answer is always an integer. "
    "End your response with exactly: Final answer: <integer>"
)
DEFAULT_SOURCES = [
    "data/external/external_math_10000.jsonl",
    "data/processed/hybrid_verbose_distill.jsonl",
]


def norm_q(q: str) -> str:
    return re.sub(r"\s+", " ", (q or "").strip().lower())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sources", nargs="+", default=DEFAULT_SOURCES)
    ap.add_argument("--output", default="data/processed/unified_prompt_13k.jsonl")
    ap.add_argument("--max-seq-length", type=int, default=2048)
    ap.add_argument("--model-path", default="Qwen/Qwen2.5-3B-Instruct")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model_path)

    # 누수 방지: holdout / filtered id 는 애초에 소스에 없어야 하지만 재확인한다.
    banned = set()
    for p in ("data/train_filtered_ids.csv", "data/holdout/official_holdout_464_clean.csv"):
        f = ROOT / p
        if f.exists():
            for line in f.read_text().splitlines()[1:]:
                banned.add(line.split(",")[0].strip())
    print(f"제외 id 목록 {len(banned)}개 로드")

    seen_q, seen_id = set(), set()
    kept, stats = [], {"dup_q": 0, "dup_id": 0, "banned": 0, "too_long": 0, "changed_prompt": 0}

    for src in args.sources:
        n_src = 0
        with open(ROOT / src) as f:
            for line in f:
                d = json.loads(line)
                rid = d.get("id")
                if rid in banned:
                    stats["banned"] += 1
                    continue
                if rid in seen_id:
                    stats["dup_id"] += 1
                    continue
                q = norm_q(d.get("question", ""))
                if q in seen_q:
                    stats["dup_q"] += 1
                    continue
                msgs = d["messages"]
                if msgs[0]["role"] != "system":
                    raise SystemExit(f"unexpected roles in {src}: {[m['role'] for m in msgs]}")
                if msgs[0]["content"] != INFERENCE_SYSTEM:
                    stats["changed_prompt"] += 1
                msgs = [{"role": "system", "content": INFERENCE_SYSTEM}] + msgs[1:]
                ids = tok.apply_chat_template(msgs, tokenize=True, add_generation_prompt=False)
                if len(ids) > args.max_seq_length:
                    stats["too_long"] += 1
                    continue
                d = dict(d)
                d["messages"] = msgs
                d["source_file"] = src
                seen_id.add(rid)
                seen_q.add(q)
                kept.append(d)
                n_src += 1
        print(f"  {src}: {n_src}건 채택")

    out = ROOT / args.output
    with open(out, "w") as f:
        for d in kept:
            f.write(json.dumps(d, ensure_ascii=False) + "\n")

    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    lens = [len(m["messages"][-1]["content"]) for m in kept]
    lens.sort()
    manifest = {
        "output": str(args.output),
        "count": len(kept),
        "sources": args.sources,
        "system_prompt": INFERENCE_SYSTEM,
        "solution_len_mean": sum(lens) // len(lens),
        "solution_len_median": lens[len(lens) // 2],
        "sha256": sha,
        **stats,
    }
    (ROOT / (args.output + ".manifest.json")).write_text(json.dumps(manifest, indent=2))
    print(json.dumps(manifest, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
