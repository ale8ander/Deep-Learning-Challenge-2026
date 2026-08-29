"""verifier 4차 시도 dev 채점 + 문제 단위 선택 시뮬레이션.

두 층으로 판정한다:
  1층 dev 정확도 — kill switch: baseline_code_verified + 0.05 미만이면 폐기
  2층 선택 시뮬레이션 — 후보 중 P(A) 최대를 고르는 규칙이, 같은 dev 문제들에서
     (a) 단순 plurality (b) 코드검증 plurality 를 이기는가.
     **이게 진짜 배포 성능이다** — dev 정확도가 높아도 문제 내 순위가 안 갈리면 무용지물.
"""
import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

DEFAULT_MODEL_PATH = "/workspace/models/Qwen2.5-3B-Instruct"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dev", type=Path, required=True)
    ap.add_argument("--adapter-path", type=Path, required=True)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    rows = [json.loads(l) for l in open(args.dev, encoding="utf-8") if l.strip()]

    tok = AutoTokenizer.from_pretrained(DEFAULT_MODEL_PATH, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    a_id = tok("A", add_special_tokens=False)["input_ids"][0]
    b_id = tok("B", add_special_tokens=False)["input_ids"][0]

    model = AutoModelForCausalLM.from_pretrained(
        DEFAULT_MODEL_PATH, local_files_only=True, torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="cuda")
    model = PeftModel.from_pretrained(model, str(args.adapter_path), local_files_only=True)
    model.eval()

    prompts = [tok.apply_chat_template(r["messages"][:-1], tokenize=False,
                                       add_generation_prompt=True) for r in rows]
    scores = []
    with torch.inference_mode():
        for i in tqdm(range(0, len(prompts), args.batch_size)):
            batch = tok(prompts[i:i + args.batch_size], return_tensors="pt",
                        padding=True, truncation=True, max_length=2048).to("cuda")
            logits = model(**batch).logits[:, -1, :]
            pair = logits[:, [a_id, b_id]].float()
            p_a = torch.softmax(pair, dim=-1)[:, 0]
            scores.extend(p_a.tolist())

    # ── 1층: dev 정확도 ──
    correct = sum(1 for r, s in zip(rows, scores)
                  if (s >= 0.5) == r["meta"]["correct"])
    acc = correct / len(rows)

    # ── 2층: 문제 단위 선택 시뮬레이션 ──
    by_pid = defaultdict(list)
    for r, s in zip(rows, scores):
        by_pid[r["meta"]["id"]].append((s, r["meta"]))
    sel = Counter()
    for pid, cands in by_pid.items():
        golds = {m["prediction"] for _, m in cands if m["correct"]}
        if not golds:
            sel["no_oracle"] += 1        # 어떤 규칙도 못 맞힘 — 비교 제외
            continue
        sel["scored"] += 1
        # verifier: P(A) 최대 후보
        v_pick = max(cands, key=lambda t: t[0])[1]["prediction"]
        sel["verifier"] += int(v_pick in golds)
        # plurality (동률이면 실패 처리)
        c = Counter(m["prediction"] for _, m in cands)
        top = c.most_common()
        p_pick = top[0][0] if (len(top) == 1 or top[0][1] > top[1][1]) else None
        sel["plurality"] += int(p_pick in golds)
        # 코드검증 plurality
        cv = Counter(m["prediction"] for _, m in cands if m["code_verified"])
        topv = cv.most_common()
        cv_pick = topv[0][0] if topv and (len(topv) == 1 or topv[0][1] > topv[1][1]) else None
        sel["code_verified_rule"] += int(cv_pick in golds)
        # verifier 를 코드검증 위에 얹기: 코드검증 후보들 중 P(A) 최대
        cvc = [(s, m) for s, m in cands if m["code_verified"]]
        hy_pick = (max(cvc, key=lambda t: t[0])[1]["prediction"] if cvc
                   else max(cands, key=lambda t: t[0])[1]["prediction"])
        sel["hybrid_cv_then_verifier"] += int(hy_pick in golds)

    result = {
        "dev_candidates": len(rows),
        "dev_accuracy": round(acc, 4),
        "selection_sim": {
            "problems_scored": sel["scored"],
            "no_oracle_excluded": sel["no_oracle"],
            "verifier_top1": sel["verifier"],
            "plurality": sel["plurality"],
            "code_verified_rule": sel["code_verified_rule"],
            "hybrid_cv_then_verifier": sel["hybrid_cv_then_verifier"],
        },
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(result, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
