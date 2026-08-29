"""NuminaMath-TIR을 우리 TIR 하네스의 2라운드 포맷으로 변환해 SFT 데이터를 만든다.

원본은 단일 assistant 턴에 [추론 → ```python → ```output → 추론 → \boxed{답}]이 인터리브된 형태다.
우리 추론 하네스(scripts/tir_inference*.py)는 2라운드다:
    system(TIR_SYSTEM) / user(문제) -> assistant(추론 + ```python 블록)
    user("Program output: ...") -> assistant(최종 추론 + "Final answer: N")

**학습과 추론의 포맷이 어긋나면 안 된다** — 이 프로젝트는 오전 세션에서 학습/추론 system
프롬프트 불일치 결함을 이미 한 번 발견했다. 그래서 여기서는 하네스의 상수를 직접 import해서
쓴다(복붙 금지, 프롬프트가 바뀌면 자동으로 따라간다).

필터:
  1. python 블록 정확히 1개, output 블록 정확히 1개 (2라운드로 깔끔히 쪼개짐)
  2. 최종 \boxed{} 답이 정수 (대회 조건: 정답은 항상 정수)
  3. 코드를 **실제로 재실행**해 출력이 정답과 일치 (teacher 라벨을 그대로 믿지 않는다)
  4. 공식 train / 리더보드 831과 정규화 문제 본문 중복 제거 (누수 방지)
  5. 토큰 길이 상한
"""
import argparse
import csv
import hashlib
import json
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tir_inference import TIR_SYSTEM, FINAL_NUDGE, run_code, normalize  # noqa: E402

CODE_BLOCK = re.compile(r"```python\s*\n(.*?)```", re.S)
OUTPUT_BLOCK = re.compile(r"```output\s*\n(.*?)```", re.S)
BOXED = re.compile(r"\\boxed\{([^{}]*)\}")


def norm_text(s):
    return re.sub(r"\s+", " ", str(s)).strip().lower()


def build_example(row, exec_timeout):
    """원본 1건 -> (SFT 메시지, 메타) 또는 None."""
    assistant = next((m["content"] for m in row["messages"] if m["role"] == "assistant"), None)
    user = next((m["content"] for m in row["messages"] if m["role"] == "user"), None)
    if not assistant or not user:
        return None

    codes = CODE_BLOCK.findall(assistant)
    outputs = OUTPUT_BLOCK.findall(assistant)
    if len(codes) != 1 or len(outputs) != 1:
        return None

    boxed = BOXED.findall(assistant)
    if not boxed:
        return None
    answer = normalize(boxed[-1].strip())
    if answer is None:
        return None

    # round 1 = 첫 코드 블록까지. 원본 output 블록 이후는 round 2로 간다.
    code_end = assistant.index("```", assistant.index(codes[0])) + 3
    round1 = assistant[:code_end].strip()
    tail = assistant[code_end:]
    out_end = tail.index("```", tail.index(outputs[0])) + 3
    round2 = tail[out_end:].strip()
    if not round2:
        return None

    # 최종 답 표기를 우리 규약으로 통일한다.
    round2 = f"{round2}\n\nFinal answer: {answer}"

    return {
        "code": codes[0],
        "claimed_output": outputs[0].strip(),
        "answer": answer,
        "round1": round1,
        "round2": round2,
        "question": user.strip(),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", type=Path, default=ROOT / "data/external/numinamath_tir_raw.jsonl")
    ap.add_argument("--output", type=Path, default=ROOT / "data/processed/numina_tir_sft.jsonl")
    ap.add_argument("--target", type=int, default=3000)
    ap.add_argument("--scan-limit", type=int, default=20000, help="원본에서 훑을 최대 행수")
    ap.add_argument("--exec-timeout", type=int, default=20)
    ap.add_argument("--exec-workers", type=int, default=96)
    ap.add_argument("--max-chars", type=int, default=6000, help="round1+round2 길이 상한")
    ap.add_argument("--style", choices=("any", "executable"), default="any",
                    help="executable: 우리 3B가 실제로 성공시키는 코드 스타일만 남긴다. "
                         "Numina는 sympy 58%인데 우리 모델 성공 코드는 12%뿐이라, "
                         "그대로 학습시키면 소화 못 하는 스타일을 배워 실행 에러가 늘었다(56->74).")
    ap.add_argument("--max-code-chars", type=int, default=700, help="style=executable 일 때 코드 길이 상한")
    ap.add_argument("--seed", type=int, default=20260828)
    args = ap.parse_args()

    # 누수 방지용 기존 문제 본문
    known = set()
    for name in ("data/deep_chal_math_train.csv", "data/deep_chal_math_leaderboard_filtered.csv"):
        path = ROOT / name
        if path.exists():
            for r in csv.DictReader(open(path, encoding="utf-8-sig")):
                known.add(norm_text(r["question"]))
    print(f"누수 대조용 기존 문제 본문 {len(known)}건 로드")

    candidates, scanned = [], 0
    for line in open(args.input):
        scanned += 1
        if scanned > args.scan_limit:
            break
        ex = build_example(json.loads(line), args.exec_timeout)
        if ex is None:
            continue
        if len(ex["round1"]) + len(ex["round2"]) > args.max_chars:
            continue
        if norm_text(ex["question"]) in known:
            continue
        if args.style == "executable":
            code = ex["code"]
            if "sympy" in code or "symbols(" in code:
                continue  # 3B가 안정적으로 못 쓰는 기호계산 스타일 배제
            if len(code) > args.max_code_chars:
                continue
        candidates.append(ex)
    print(f"원본 {scanned}건 훑음 -> 구조·정수답·길이·누수 필터 통과 {len(candidates)}건")

    # 정렬을 고정해 재현 가능하게 만든 뒤, 코드 재실행 검증은 목표치의 1.6배만 시도한다.
    candidates.sort(key=lambda e: int(hashlib.sha256(f"{args.seed}:{e['question']}".encode()).hexdigest()[:12], 16))
    trial = candidates[: int(args.target * 1.6)]
    print(f"코드 재실행 검증 대상 {len(trial)}건 (워커 {args.exec_workers}, 타임아웃 {args.exec_timeout}s)")

    with ThreadPoolExecutor(max_workers=args.exec_workers) as pool:
        results = list(pool.map(lambda e: run_code(e["code"], args.exec_timeout), trial))

    verified, stats = [], {"ok": 0, "error": 0, "timeout": 0, "mismatch": 0}
    for ex, res in zip(trial, results):
        stats[res["status"]] = stats.get(res["status"], 0) + 1
        if res["status"] != "ok":
            continue
        produced = normalize(res["stdout"].strip().splitlines()[-1] if res["stdout"].strip() else None)
        if produced is None or produced != ex["answer"]:
            stats["mismatch"] += 1
            continue
        ex["real_output"] = res["stdout"].strip()
        verified.append(ex)
        if len(verified) >= args.target:
            break

    print(f"재실행 결과: 성공 {stats['ok']} / 에러 {stats['error']} / 타임아웃 {stats['timeout']}"
          f" / 답 불일치 {stats['mismatch']}")
    print(f"최종 검증 통과 {len(verified)}건")

    # 하네스와 완전히 동일한 2라운드 대화로 직렬화
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for i, ex in enumerate(verified):
            feedback = (
                f"Program output:\n```\n{ex['real_output']}\n```\n{FINAL_NUDGE}"
            )
            messages = [
                {"role": "system", "content": TIR_SYSTEM},
                {"role": "user", "content": ex["question"]},
                {"role": "assistant", "content": ex["round1"]},
                {"role": "user", "content": feedback},
                {"role": "assistant", "content": ex["round2"]},
            ]
            f.write(json.dumps({"id": f"numina-tir-{i:06d}", "messages": messages,
                                "answer": ex["answer"]}, ensure_ascii=False) + "\n")

    digest = hashlib.sha256(args.output.read_bytes()).hexdigest()
    manifest = {
        "output": str(args.output if not str(args.output).startswith(str(ROOT)) else args.output.relative_to(ROOT)),
        "sha256": digest,
        "count": len(verified),
        "source": "AI-MO/NuminaMath-TIR (Apache-2.0, 공개 데이터)",
        "scanned": scanned,
        "structure_filtered": len(candidates),
        "exec_verified": len(verified),
        "exec_stats": stats,
        "filters": [
            "python 블록 1개 + output 블록 1개",
            "최종 boxed 답이 정수",
            "코드 재실행 출력이 답과 일치",
            "공식 train/리더보드 831과 문제 본문 중복 제거",
            f"round1+round2 길이 <= {args.max_chars}자",
        ],
        "prompt_source": "scripts/tir_inference.py 의 TIR_SYSTEM / FINAL_NUDGE 직접 import",
        "seed": args.seed,
    }
    with open(str(args.output).replace(".jsonl", ".manifest.json"), "w") as f:
        json.dump(manifest, f, ensure_ascii=False, indent=2)
    print(f"저장: {args.output}  sha256={digest[:16]}...")


if __name__ == "__main__":
    main()
