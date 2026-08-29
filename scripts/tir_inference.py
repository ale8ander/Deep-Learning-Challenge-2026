"""TIR (tool-integrated reasoning) 스모크 — 모델이 짠 파이썬을 로컬에서 실행해 되먹인다.

⚠️ 규정 상태: 추론 시점 코드 실행은 대회 규정상 회색지대다(금지 3개 조항에는 안 걸리지만
허용 목록은 학습 기법 열거이고, 추론 자유 조항의 예시는 전부 샘플링 계열이다).
**운영진 확인 전까지 이 스크립트의 산출물로 리더보드에 제출하지 않는다.** 연구용 측정 전용.

동작:
  round 1: 문제 + "파이썬으로 계산하라" 프롬프트 -> 모델이 ```python 블록 생성
  실행:    subprocess 격리, 타임아웃, stdout 캡처
  round 2: 실행 결과를 대화에 되먹임 -> 모델이 최종 답 (Final answer: N)
  코드가 없거나 실행이 실패하면 그냥 모델이 이어서 풀게 둔다(순수 CoT로 열화).

기존 baseline.py와 같은 답 추출·채점 규약을 쓴다.
"""
import argparse
import csv
import json
import re
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import torch
from tqdm import tqdm
from transformers import AutoModelForCausalLM, AutoTokenizer

DEFAULT_MODEL_PATH = Path("/workspace/models/Qwen2.5-3B-Instruct")

# 상수·순수 함수는 scripts/tir_common.py 로 이동했다 (클라이언트가 torch 를 끌지 않도록).
# 여기서 re-export 하므로 `from tir_inference import TIR_SYSTEM, ...` 은 그대로 동작한다.
from tir_common import (  # noqa: F401,E402
    TIR_SYSTEM, FINAL_NUDGE, CODE_BLOCK, ANSWER_PATTERNS,
    SANDBOX_PREAMBLE, extract_answer, normalize, run_code,
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--adapter-path", type=str, default=None)
    parser.add_argument("--model-path", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--max-new-tokens", type=int, default=768)
    parser.add_argument("--final-max-new-tokens", type=int, default=512)
    parser.add_argument("--exec-timeout", type=int, default=10)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]

    tokenizer = AutoTokenizer.from_pretrained(args.model_path, local_files_only=True)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model_path, dtype=torch.float16, local_files_only=True
    )
    if args.adapter_path:
        from peft import PeftModel
        model = PeftModel.from_pretrained(model, args.adapter_path, local_files_only=True)
    model = model.cuda().eval()

    args.output.parent.mkdir(parents=True, exist_ok=True)
    started = time.time()
    stats = {"code_found": 0, "exec_ok": 0, "exec_error": 0, "exec_timeout": 0, "correct": 0, "total": 0}

    with args.output.open("w", encoding="utf-8") as out:
        for start in tqdm(range(0, len(rows), args.batch_size), desc="tir"):
            batch = rows[start : start + args.batch_size]
            convos = [
                [
                    {"role": "system", "content": TIR_SYSTEM},
                    {"role": "user", "content": row["question"]},
                ]
                for row in batch
            ]
            prompts = [
                tokenizer.apply_chat_template(c, tokenize=False, add_generation_prompt=True)
                for c in convos
            ]
            first = generate(model, tokenizer, prompts, args.max_new_tokens)

            # 코드 추출 + 실행
            executions = []
            for text in first:
                blocks = CODE_BLOCK.findall(text)
                if not blocks:
                    executions.append(None)
                    continue
                stats["code_found"] += 1
                result = run_code(blocks[-1], args.exec_timeout)
                stats[f"exec_{result['status']}"] = stats.get(f"exec_{result['status']}", 0) + 1
                executions.append(result)

            # 2라운드 프롬프트 구성
            second_prompts = []
            for convo, text, execution in zip(convos, first, executions):
                if execution is None:
                    feedback = (
                        "No Python code block was found in your response. "
                        "Solve the problem directly and end with exactly: Final answer: <integer>"
                    )
                else:
                    body = execution["stdout"].strip() or "(no output)"
                    if execution["status"] != "ok":
                        body += f"\n[{execution['status']}] {execution['stderr'].strip()[:400]}"
                    feedback = f"Program output:\n```\n{body}\n```\n{FINAL_NUDGE}"
                new_convo = convo + [
                    {"role": "assistant", "content": text},
                    {"role": "user", "content": feedback},
                ]
                second_prompts.append(
                    tokenizer.apply_chat_template(new_convo, tokenize=False, add_generation_prompt=True)
                )
            second = generate(model, tokenizer, second_prompts, args.final_max_new_tokens)

            for row, text1, execution, text2 in zip(batch, first, executions, second):
                prediction = normalize(extract_answer(text2))
                if prediction is None:
                    prediction = normalize(extract_answer(text1))
                answer = normalize(row.get("answer"))
                correct = prediction is not None and prediction == answer
                stats["total"] += 1
                stats["correct"] += int(correct)
                out.write(json.dumps({
                    "id": row["id"],
                    "answer": answer,
                    "prediction": prediction,
                    "correct": correct,
                    "code_executed": execution is not None,
                    "exec_status": execution["status"] if execution else None,
                    "exec_stdout": execution["stdout"][:500] if execution else None,
                    "round1": text1,
                    "round2": text2,
                }, ensure_ascii=False) + "\n")
                out.flush()

    elapsed = time.time() - started
    stats["elapsed_sec"] = round(elapsed, 1)
    print(json.dumps(stats, ensure_ascii=False))
    print(f"correct={stats['correct']}/{stats['total']}  elapsed={elapsed:.0f}s -> {args.output}")


if __name__ == "__main__":
    main()
