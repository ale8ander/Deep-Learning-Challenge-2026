"""역검증 (answer-conditioned verification) — 후보를 '내용'으로 거른다.

왜 이걸 하는가
--------------
표수≤3 구간에서 정답이 후보 안에 있는데 못 고르는 문제가 홀드아웃 87문제 중 14개다
(코드검증 오라클 45, 실채택 31). CONTEXT 21절에서 표를 세는 신호는 전부 소진됐다 —
margin, breadth, 계보 다양성, 코드 다양성 넷 다 전환율 0이었다. 남은 건 후보의 내용이다.

현행 TIR 게이트("exec ok + stdout 정수 + stdout == 최종답")는 **정확성이 아니라 일관성**
검사다. 같은 잘못된 접근을 8번 실행해도 통과한다.

이 하네스는 방향을 뒤집는다. 답을 **주고** 그 답이 문제의 조건을 만족하는지 검사하는
프로그램을 짜게 한다.
  - 생성 문제(답을 찾아라)가 아니라 결정 문제(이 답이 맞나)다. 3B에게 난이도가 다르다.
  - 이 대회 미해결 문제는 number_theory + combinatorics 가 54%이고, 그 유형은 답을 넣고
    조건을 확인하는 게 브루트포스 한 줄이다. 이 모델은 이미 성공 코드의 30%가 브루트포스다.
  - 학습형 selector 3연패 및 LLM 심판과 범주가 다르다. **판정 주체가 모델이 아니라
    인터프리터다.** 모델은 검사 코드를 짜는 역할만 하고 채택은 프로그램 출력이 정한다.

⚠️ 추론 시점 코드 실행은 대회 규정 회색지대다. CONTEXT "TIR" 절 참고. 백업 제출본 유지.
"""
import argparse
import csv
import json
import re
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tir_common import CODE_BLOCK, normalize, run_code  # noqa: E402

VERIFY_SYSTEM = (
    "You are given a math problem and a CANDIDATE answer. Your job is to test whether the "
    "candidate answer is correct — not to solve the problem from scratch.\n"
    "Write ONE Python program that checks the candidate INDEPENDENTLY. Good checks are: "
    "brute-force enumeration over the relevant range, substituting the candidate back into "
    "every stated condition, or recomputing the quantity by a different route.\n"
    "The program must print exactly one line and nothing else:\n"
    "  VERIFIED   if the candidate satisfies every condition of the problem\n"
    "  REFUTED <n>  if the check fails, where <n> is the integer your check produced "
    "(write REFUTED alone if you cannot produce one)\n"
    "Use only the standard library and sympy. Put the program in a single ```python block."
)

VERDICT = re.compile(r"\b(VERIFIED|REFUTED)\b")


def read_rows(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in open(p) if line.strip()]
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def load_candidates(paths, extra):
    """문제별 후보 답 목록. TIR 코드검증 답 + 지정한 추가 답(보통 baseline)."""
    cand = {}
    for path in paths:
        p = ROOT / path if not Path(path).is_absolute() else Path(path)
        if not p.exists():
            print(f"[경고] 후보 파일 없음, 건너뜀: {p}")
            continue
        for r in read_rows(p):
            c = cand.setdefault(r["id"], Counter())
            for k, v in (r.get("verified_counts") or {}).items():
                k = normalize(k)
                if k is not None:
                    c[k] += v
            for k in (r.get("sample_predictions") or []):
                k = normalize(k)
                if k is not None:
                    c[k] += 0  # 후보로만 등록, 표는 주지 않음
    for pid, answer in (extra or {}).items():
        if answer is not None:
            cand.setdefault(pid, Counter()).setdefault(answer, 0)
    return cand


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True, help="문제 CSV/JSONL (id, question[, answer])")
    ap.add_argument("--candidates", nargs="+", required=True, help="verified_counts 를 가진 jsonl")
    ap.add_argument("--baseline", default=None, help="baseline_prediction 을 가진 jsonl (후보에 추가)")
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="hybrid3145")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--max-candidates", type=int, default=4)
    ap.add_argument("--num-samples", type=int, default=4, help="후보당 검사 프로그램 샘플 수")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=1024)
    ap.add_argument("--exec-timeout", type=int, default=30)
    ap.add_argument("--exec-workers", type=int, default=32)
    ap.add_argument("--request-workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)

    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]

    extra = {}
    if args.baseline:
        for r in read_rows(args.baseline):
            extra[r["id"]] = normalize(r.get("baseline_prediction"))
    cand_map = load_candidates(args.candidates, extra)

    # (문제, 후보) 작업 목록. 후보가 1개뿐이면 비교할 게 없으므로 건너뛴다.
    jobs = []
    for r in rows:
        c = cand_map.get(r["id"], Counter())
        top = [a for a, _ in c.most_common(args.max_candidates)]
        if len(top) < 2:
            continue
        for a in top:
            jobs.append((r, a))
    print(f"문제 {len(rows)} / 검증 대상 {len(set(j[0]['id'] for j in jobs))} / "
          f"(문제,후보) 쌍 {len(jobs)} / 샘플 {args.num_samples} "
          f"=> 생성 {len(jobs)*args.num_samples}회")

    N = args.num_samples

    def chat(messages):
        resp = client.chat.completions.create(
            model=args.model, messages=messages, max_tokens=args.max_new_tokens,
            n=N, seed=args.seed, temperature=args.temperature, top_p=args.top_p)
        texts = [c.message.content or "" for c in sorted(resp.choices, key=lambda c: c.index)]
        return texts + [""] * (N - len(texts))

    started = time.time()
    convos = [
        [{"role": "system", "content": VERIFY_SYSTEM},
         {"role": "user", "content": f"{r['question']}\n\nCANDIDATE ANSWER: {a}\n\n"
                                     f"Write the checking program."}]
        for r, a in jobs
    ]
    with ThreadPoolExecutor(max_workers=args.request_workers) as pool:
        per_job = list(pool.map(chat, convos))
    gen_sec = time.time() - started

    flat = [t for texts in per_job for t in texts]
    blocks = [CODE_BLOCK.findall(t) for t in flat]
    exec_started = time.time()
    with ThreadPoolExecutor(max_workers=args.exec_workers) as pool:
        futures = [pool.submit(run_code, b[-1], args.exec_timeout) if b else None for b in blocks]
        executions = [f.result() if f is not None else None for f in futures]
    exec_sec = time.time() - exec_started

    # (문제,후보) 별 판정 집계
    results = {}
    stats = Counter()
    for j, (r, a) in enumerate(jobs):
        tally = Counter()
        refuted_to = Counter()
        for k in range(N):
            ex = executions[j * N + k]
            if ex is None:
                tally["no_code"] += 1
                continue
            stats[f"exec_{ex['status']}"] += 1
            if ex["status"] != "ok":
                tally["exec_fail"] += 1
                continue
            out = (ex["stdout"] or "").strip()
            m = VERDICT.search(out)
            if not m:
                tally["no_verdict"] += 1
                continue
            tally[m.group(1)] += 1
            if m.group(1) == "REFUTED":
                tail = normalize(out[m.end():].strip().split()[0]) if out[m.end():].strip() else None
                if tail is not None:
                    refuted_to[tail] += 1
        results.setdefault(r["id"], {"row": r, "cands": {}})
        results[r["id"]]["cands"][a] = {"tally": dict(tally), "refuted_to": dict(refuted_to)}
        stats["pairs"] += 1

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for pid, d in results.items():
            rec = {"id": pid, "answer": normalize(d["row"].get("answer")),
                   "candidates": {str(k): v for k, v in d["cands"].items()}}
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    stats["gen_sec"] = round(gen_sec, 1)
    stats["exec_sec"] = round(exec_sec, 1)
    stats["total_sec"] = round(time.time() - started, 1)
    print(json.dumps(dict(stats), ensure_ascii=False))
    print(f"-> {args.output}")


if __name__ == "__main__":
    main()
