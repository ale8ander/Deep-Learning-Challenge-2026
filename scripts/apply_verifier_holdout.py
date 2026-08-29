"""학습된 verifier 를 홀드아웃 TIR 궤적에 적용해 **챔피언 대비** 판정한다.

dev 통과(73.8%)는 필요조건일 뿐이다. 이 프로젝트는 홀드아웃→Public 3연패,
오라클→실채택 전환 실패 6회 전력이 있다. 진짜 질문은 하나다:

    현재 배포 규칙(코드검증 plurality + min-count)에 verifier 를 얹으면
    **챔피언 대비 순이득이 늘어나는가?**

그래서 대체가 아니라 **재순위(re-ranking)** 로만 쓴다:
  코드검증 통과 후보들 중에서 P(A) 최고를 고른다. 코드검증 후보가 없으면 발동하지 않는다.
이 구조면 기존 신호를 버리지 않으므로 하방이 작다 (dev 시뮬에서도 이 조합이 최고였다).

출력은 gain/regression/발동수 + calibration/validation 분할이다.
⚠️ 87문제 표준오차는 ±3.5~3.9다 (CONTEXT C-3). ±4 미만 차이를 실효과로 읽지 말 것.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

import torch
from peft import PeftModel
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tir_common import normalize  # noqa: E402
from build_verifier_tir_data import VERIFIER_SYSTEM, trim  # noqa: E402

BASE = "/workspace/models/Qwen2.5-3B-Instruct"


def half(pid):
    """CONTEXT 관행대로 문제를 calibration/validation 으로 반 가른다."""
    h = int(hashlib.sha256(f"split:{pid}".encode()).hexdigest()[:8], 16)
    return "calib" if h % 2 == 0 else "valid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--traj", type=Path, required=True, help="홀드아웃 궤적 덤프")
    ap.add_argument("--questions", type=Path,
                    default=ROOT / "data/holdout/holdout464_vote3.csv")
    ap.add_argument("--adapter-path", type=Path, required=True)
    ap.add_argument("--champion", type=Path, required=True,
                    help="챔피언 홀드아웃 등가물 (compose_holdout464.py 산출) 또는 id->answer jsonl")
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import csv
    qs = {r["id"]: r["question"] for r in csv.DictReader(
        open(args.questions, encoding="utf-8-sig"))}
    gold = {r["id"]: normalize(r.get("answer")) for r in csv.DictReader(
        open(args.questions, encoding="utf-8-sig"))}

    champ = {}
    for line in open(args.champion, encoding="utf-8"):
        if line.strip():
            r = json.loads(line)
            champ[r["id"]] = normalize(r.get("prediction"))

    cands = defaultdict(list)
    for line in open(args.traj, encoding="utf-8"):
        r = json.loads(line)
        p = normalize(r.get("prediction"))
        if p is None or r["id"] not in qs:
            continue
        so = normalize((r.get("exec_stdout") or "").strip())
        cands[r["id"]].append({
            "pred": p,
            "code_verified": r.get("exec_status") == "ok" and so is not None and so == p,
            "text": r.get("latest_text"), "stdout": r.get("exec_stdout"),
            "status": r.get("exec_status"),
        })

    tok = AutoTokenizer.from_pretrained(BASE, local_files_only=True)
    tok.padding_side = "left"
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    a_id = tok("A", add_special_tokens=False)["input_ids"][0]
    b_id = tok("B", add_special_tokens=False)["input_ids"][0]
    model = AutoModelForCausalLM.from_pretrained(
        BASE, local_files_only=True, torch_dtype=torch.bfloat16,
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_quant_type="nf4"),
        device_map="cuda")
    model = PeftModel.from_pretrained(model, str(args.adapter_path), local_files_only=True)
    model.eval()

    flat = [(pid, i) for pid in cands for i in range(len(cands[pid]))]
    prompts = []
    for pid, i in flat:
        c = cands[pid][i]
        body = (c["stdout"] or "").strip() or "(no output)"
        if c["status"] not in (None, "ok"):
            body += f"\n[{c['status']}]"
        user = (f"Problem:\n{qs[pid]}\n\nCandidate solution:\n{trim(c['text'])}\n\n"
                f"Program output:\n{trim(body, 300, 100)}\n\n"
                f"Proposed final answer: {c['pred']}\n\n"
                "Is the proposed answer correct? Reply with exactly one letter: "
                "A (correct) or B (incorrect).")
        prompts.append(tok.apply_chat_template(
            [{"role": "system", "content": VERIFIER_SYSTEM},
             {"role": "user", "content": user}], tokenize=False, add_generation_prompt=True))

    with torch.inference_mode():
        for s in tqdm(range(0, len(prompts), args.batch_size)):
            b = tok(prompts[s:s + args.batch_size], return_tensors="pt",
                    padding=True, truncation=True, max_length=2048).to("cuda")
            lg = model(**b).logits[:, -1, :]
            pa = torch.softmax(lg[:, [a_id, b_id]].float(), dim=-1)[:, 0].tolist()
            for (pid, i), v in zip(flat[s:s + args.batch_size], pa):
                cands[pid][i]["p"] = v

    def deployed(cs):
        """현행 배포 규칙: 코드검증 plurality, min-count 이상, 동률이면 발동 안 함."""
        c = Counter(x["pred"] for x in cs if x["code_verified"])
        t = c.most_common()
        if not t or t[0][1] < args.min_count:
            return None
        if len(t) > 1 and t[0][1] == t[1][1]:
            return None
        return t[0][0]

    def reranked(cs):
        """코드검증 후보 중 P(A) 최고. 코드검증 후보가 없으면 발동 안 함."""
        cv = [x for x in cs if x["code_verified"]]
        return max(cv, key=lambda x: x["p"])["pred"] if cv else None

    res = {}
    for name, rule in (("deployed", deployed), ("reranked", reranked)):
        st = Counter()
        for pid, cs in cands.items():
            g, base = gold.get(pid), champ.get(pid)
            if g is None or base is None:
                continue
            pick = rule(cs)
            final = pick if pick is not None else base
            st["n"] += 1
            st[f"{half(pid)}_base"] += int(base == g)
            st[f"{half(pid)}_new"] += int(final == g)
            st["base_correct"] += int(base == g)
            st["new_correct"] += int(final == g)
            if pick is not None and pick != base:
                st["fired"] += 1
                st["gain"] += int(final == g and base != g)
                st["reg"] += int(final != g and base == g)
        res[name] = {
            "n": st["n"], "champion_correct": st["base_correct"],
            "rule_correct": st["new_correct"],
            "delta": st["new_correct"] - st["base_correct"],
            "fired": st["fired"], "gain": st["gain"], "regression": st["reg"],
            "calib_delta": st["calib_new"] - st["calib_base"],
            "valid_delta": st["valid_new"] - st["valid_base"],
        }

    # 후보별 P(A) 를 남긴다 — 규칙 변형(tie-break 전용 등)을 재추론 없이 스윕하려면 필요하다.
    dump = Path(str(args.output).replace(".json", "_scored.jsonl"))
    with dump.open("w", encoding="utf-8") as f:
        for pid, cs in cands.items():
            for c in cs:
                f.write(json.dumps({"id": pid, "pred": str(c["pred"]),
                                    "code_verified": c["code_verified"],
                                    "p": c.get("p")}, ensure_ascii=False) + "\n")
    res["scored_dump"] = str(dump)

    res["note"] = ("87문제 표준오차 ±3.5~3.9 (CONTEXT C-3). "
                   "delta 차이가 4 미만이면 실효과로 읽지 말 것.")
    args.output.write_text(json.dumps(res, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(res, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
