"""TIR + 코드 리페어 + adaptive 샘플링.

기존 `tir_inference_client.py` 는 건드리지 않는다 (647 제출본 재현에 필요).

── 왜 리페어인가 ────────────────────────────────────────────────
기존 하네스의 채택 규칙은 `verified()` 이고, 여기에 `exec_status == "ok"` 가 들어 있다.
즉 **코드가 에러 난 샘플은 verified_counts 에 절대 못 들어간다 — 통째로 소각된다.**
게이트 구간 실측으로 exec_error 51~53 + timeout 11 = 샘플의 약 12% 가 이렇게 버려진다.

그런데 현재 2라운드 피드백은 에러 메시지를 보여주면서 `FINAL_NUDGE`("최종답을 내라")를
붙인다. 고칠 기회를 안 준다. 리페어는 이 12% 를 되살리는 것이다.

  라운드1: 코드 생성 -> 실행
  라운드2(리페어, 실패한 샘플만): 에러 되먹임 -> **고친 코드 요구** -> 재실행   [--repair-rounds K]
  라운드3: (고쳐진) 출력 되먹임 -> 최종답

TIR 에서 큰 이득이 났던 두 번이 전부 하네스 파라미터였다(타임아웃 10->60초가 +8을 +13으로).
라운드 수는 같은 성격의 미탐색 파라미터다.

── 왜 adaptive 인가 ─────────────────────────────────────────────
현재는 게이트 전 문제에 N=8 을 균일하게 쓴다. 그런데 표가 갈리지 않는 문제(동률/무결론)에만
계산을 더 쓰면 같은 예산으로 더 많이 회수할 수 있다.
`--adaptive-extra M` 을 주면 1차 N 샘플 뒤 **코드검증 plurality 가 확정되지 않은 문제에만**
M 샘플을 추가 생성해 합친다.

출력 포맷은 기존 하네스와 동일하다(`verified_counts`, `sample_predictions`, ...).
그래서 `rebuild_chain.py` / 기존 분석 스크립트가 그대로 먹는다.

⚠️ 추론 시점 코드 실행은 대회 규정 회색지대다. CONTEXT "TIR" 절 참고. 백업 제출본 유지.
"""
import argparse
import csv
import json
import sys
import time
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tir_common import (  # noqa: E402
    TIR_SYSTEM, TIR_SYMPY_SYSTEM, FINAL_NUDGE, CODE_BLOCK, extract_answer, normalize, run_code,
)

REPAIR_NUDGE = (
    "Your program failed to run. Read the error above, find the bug, and write a "
    "CORRECTED complete Python program in a single ```python block. "
    "Do not explain — output only the corrected program. "
    "Keep it simple: prefer a direct brute-force loop over clever library calls, "
    "and make sure it prints exactly one integer."
)

# 코드를 아예 안 쓴 샘플용. 잘림이 아니라 "산문으로 풀어버린" 경우다.
# 실측(게이트 40문제 x 8샘플): 코드 없는 샘플 22.2%, 그중 잘린 것은 1.4%뿐이고
# 오히려 코드 있는 샘플보다 길다(2426자 vs 2012자). 즉 지면이 부족한 게 아니라
# 지시를 안 따른 것이다. 코드 실행 시 정답률 22.6% vs 미실행 5.4% 이므로 회수 가치가 크다.
NOCODE_NUDGE = (
    "You did not write a Python program. Do that now.\n"
    "Output ONLY a single ```python block — no prose, no explanation.\n"
    "Prefer a direct brute-force loop over clever library calls. "
    "The program must print exactly one integer."
)


def read_rows(path):
    p = Path(path)
    if p.suffix == ".jsonl":
        return [json.loads(line) for line in open(p) if line.strip()]
    with p.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", type=Path, required=True)
    ap.add_argument("--model", default="hybrid3145")
    ap.add_argument("--base-url", default="http://localhost:8000/v1")
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--repair-rounds", type=int, default=1,
                    help="실행 실패 샘플에 코드 수정 기회를 몇 번 줄지 (0 = 기존 하네스와 동일)")
    ap.add_argument("--nocode-retries", type=int, default=0,
                    help="코드 블록을 안 쓴 샘플에 다시 요구할 횟수 (0 = 끔). "
                         "게이트 샘플의 21.8%가 코드 미생성이고 그중 잘림은 1.4%뿐이다")
    ap.add_argument("--adaptive-extra", type=int, default=0,
                    help="plurality 미확정 문제에만 추가 생성할 샘플 수 (0 = 끔)")
    ap.add_argument("--adaptive-min-count", type=int, default=2,
                    help="이 표수 이상이면 '확정'으로 보고 추가 생성하지 않는다")
    ap.add_argument("--system-style", default="default", choices=["default", "sympy"],
                    help="sympy = 대수 특화 기호 연산 프롬프트 (TIR_SYMPY_SYSTEM)")
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--top-p", type=float, default=0.95)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--final-max-new-tokens", type=int, default=1024)
    ap.add_argument("--repair-max-new-tokens", type=int, default=1024)
    ap.add_argument("--exec-timeout", type=int, default=60)
    ap.add_argument("--exec-workers", type=int, default=32)
    ap.add_argument("--request-workers", type=int, default=32)
    ap.add_argument("--seed", type=int, default=20260828)
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--exemplars", default=None,
                    help="few-shot 예시 json ([{user, assistant}]) — system 뒤에 대화 턴으로 삽입")
    ap.add_argument("--dump-trajectories", type=Path, default=None,
                    help="샘플별 전체 궤적(1라운드 텍스트/최종 코드/stdout/최종답 텍스트)을 "
                         "jsonl 로 남긴다. TIR 자기증류 SFT 수확용 — 기본 출력은 예측/상태만 "
                         "저장해서 학습 데이터로 못 쓴다")
    args = ap.parse_args()

    from openai import OpenAI
    client = OpenAI(base_url=args.base_url, api_key="EMPTY", timeout=1800)

    rows = read_rows(args.input)
    if args.limit:
        rows = rows[: args.limit]

    stats = Counter()
    traj_dump = []          # --dump-trajectories 용. generate_pool 이 호출될 때마다 쌓인다.
    t0 = time.time()

    def chat(messages, max_tokens, n, seed):
        """컨텍스트 초과(400)면 assistant 본문을 잘라 재시도한다.

        max_model_len 이 4096 인데 리페어가 턴을 늘리므로 긴 문제에서 넘칠 수 있다.
        831 게이트 실행이 실제로 이걸로 죽었다. 한 샘플 때문에 전체가 죽으면 안 된다.
        """
        kw = dict(model=args.model, messages=messages, max_tokens=max_tokens, n=n,
                  seed=seed, temperature=args.temperature, top_p=args.top_p)
        for attempt in range(3):
            try:
                resp = client.chat.completions.create(**kw)
                texts = [c.message.content or "" for c in sorted(resp.choices, key=lambda c: c.index)]
                return texts + [""] * (n - len(texts))
            except Exception as e:
                if "maximum context length" not in str(e) and "400" not in str(e):
                    raise
                stats["ctx_overflow_retry"] += 1
                kw["max_tokens"] = max(256, kw["max_tokens"] // 2)
                trimmed = []
                for m in kw["messages"]:
                    if m["role"] == "assistant" and len(m["content"]) > 1500:
                        trimmed.append({**m, "content": m["content"][-1500:]})
                    else:
                        trimmed.append(m)
                kw["messages"] = trimmed
        stats["ctx_overflow_gaveup"] += 1
        return [""] * n

    def run_batch(convos, timeout):
        """코드 블록 추출 -> 병렬 실행."""
        blocks = [CODE_BLOCK.findall(t) for t in convos]
        with ThreadPoolExecutor(max_workers=args.exec_workers) as pool:
            futs = [pool.submit(run_code, b[-1], timeout) if b else None for b in blocks]
            return [f.result() if f is not None else None for f in futs]

    def generate_pool(rows_subset, n, seed):
        """문제별 n샘플 생성 -> 리페어 -> 최종답. (sample_pred, execution) 리스트를 문제별로 돌려준다."""
        sys_prompt = TIR_SYMPY_SYSTEM if args.system_style == "sympy" else TIR_SYSTEM
        shot_turns = []
        if args.exemplars:
            for s in json.load(open(args.exemplars)):
                shot_turns += [{"role": "user", "content": s["user"]},
                               {"role": "assistant", "content": s["assistant"]}]
        base = [[{"role": "system", "content": sys_prompt}, *shot_turns,
                 {"role": "user", "content": r["question"]}] for r in rows_subset]
        with ThreadPoolExecutor(max_workers=args.request_workers) as pool:
            per_row = list(pool.map(lambda c: chat(c, args.max_new_tokens, n, seed), base))
        first = [t for texts in per_row for t in texts]           # idx = r*n + k
        execs = run_batch(first, args.exec_timeout)
        # 대화 상태를 샘플별로 들고 다닌다 (리페어가 여러 턴이 될 수 있다).
        # `latest_code` 는 최종 라운드용 압축 대화를 만들 때 쓴다 — 실패한 시도와 리페어 지시를
        # 전부 이어붙이면 max_model_len(4096)을 넘겨 400 이 난다(831 게이트에서 실제로 터졌다).
        convo = [base[i // n] + [{"role": "assistant", "content": first[i]}]
                 for i in range(len(first))]
        latest = list(first)
        from_nocode = [False] * len(first)   # 궤적 출처 기록 (덤프용)
        from_repair = [False] * len(first)
        for i, e in enumerate(execs):
            stats["r1_no_code"] += int(e is None)
            if e is not None:
                stats[f"r1_{e['status']}"] += 1

        # ── 코드 미생성 재시도 ──
        # execs[i] is None 은 "코드 블록이 없었다"는 뜻이고, 그런 샘플은 verified_counts 에
        # 절대 못 들어가 통째로 버려진다(게이트 샘플의 21.8%). 한 번 더 요구한다.
        for rd in range(args.nocode_retries):
            todo = [i for i, e in enumerate(execs) if e is None]
            if not todo:
                break
            stats[f"nocode{rd+1}_attempted"] += len(todo)
            msgs = [convo[i] + [{"role": "user", "content": NOCODE_NUDGE}] for i in todo]
            with ThreadPoolExecutor(max_workers=args.request_workers) as pool:
                got = [t[0] for t in pool.map(
                    lambda c: chat(c, args.repair_max_new_tokens, 1, seed + 2000 + rd), msgs)]
            new_execs = run_batch(got, args.exec_timeout)
            for j, i in enumerate(todo):
                if new_execs[j] is None:
                    continue                      # 여전히 코드 없음 — 그대로 둔다
                convo[i] = msgs[j] + [{"role": "assistant", "content": got[j]}]
                latest[i] = got[j]
                execs[i] = new_execs[j]
                from_nocode[i] = True
                stats[f"nocode{rd+1}_produced"] += 1
                if new_execs[j]["status"] == "ok":
                    stats[f"nocode{rd+1}_ok"] += 1

        # ── 리페어 라운드 ──
        for rd in range(args.repair_rounds):
            todo = [i for i, e in enumerate(execs)
                    if e is not None and e["status"] != "ok"]
            if not todo:
                break
            stats[f"repair{rd+1}_attempted"] += len(todo)
            msgs = []
            for i in todo:
                err = (execs[i]["stderr"] or "").strip()[:400] or execs[i]["status"]
                msgs.append(convo[i] + [{"role": "user",
                                         "content": f"Program output:\n```\n[{execs[i]['status']}] {err}\n```\n{REPAIR_NUDGE}"}])
            with ThreadPoolExecutor(max_workers=args.request_workers) as pool:
                fixed = [t[0] for t in pool.map(
                    lambda c: chat(c, args.repair_max_new_tokens, 1, seed + 1000 + rd), msgs)]
            new_execs = run_batch(fixed, args.exec_timeout)
            for j, i in enumerate(todo):
                convo[i] = msgs[j] + [{"role": "assistant", "content": fixed[j]}]
                latest[i] = fixed[j]
                from_repair[i] = True
                if new_execs[j] is not None:
                    if new_execs[j]["status"] == "ok":
                        stats[f"repair{rd+1}_fixed"] += 1
                    execs[i] = new_execs[j]

        # ── 최종 라운드 ──
        finals = []
        for i in range(len(first)):
            e = execs[i]
            if e is None:
                fb = ("No Python code block was found. Solve the problem directly and end "
                      "with exactly: Final answer: <integer>")
            else:
                body = (e["stdout"] or "").strip() or "(no output)"
                if e["status"] != "ok":
                    body += f"\n[{e['status']}] {(e['stderr'] or '').strip()[:400]}"
                fb = f"Program output:\n```\n{body}\n```\n{FINAL_NUDGE}"
            # 최종 라운드는 **압축된 대화**를 쓴다: system + 문제 + (마지막으로 시도한) 코드 + 그 출력.
            # 실패한 시도와 리페어 지시는 최종답에 필요 없고, 이어붙이면 컨텍스트를 넘긴다.
            finals.append(base[i // n] + [{"role": "assistant", "content": latest[i]},
                                          {"role": "user", "content": fb}])
        with ThreadPoolExecutor(max_workers=args.request_workers) as pool:
            second = [t[0] for t in pool.map(
                lambda c: chat(c, args.final_max_new_tokens, 1, seed + 7), finals)]

        out = []
        for r in range(len(rows_subset)):
            per = []
            for k in range(n):
                i = r * n + k
                pred = normalize(extract_answer(second[i])) or normalize(extract_answer(first[i]))
                per.append((pred, execs[i]))
                if args.dump_trajectories is not None:
                    e = execs[i]
                    traj_dump.append({
                        "id": rows_subset[r]["id"],
                        "answer": rows_subset[r].get("answer"),
                        "sample_index": k,
                        "prediction": None if pred is None else str(pred),
                        "exec_status": None if e is None else e["status"],
                        "exec_stdout": None if e is None else e.get("stdout"),
                        # latest = 코드가 실제로 실행된 assistant 턴 (리페어/재시도 반영본).
                        # SFT 는 이걸 1라운드 응답으로 "역사 재작성"해 쓴다 — 첫 시도에
                        # 돌아가는 코드를 내는 행동 자체를 가르치는 게 목적이기 때문이다.
                        "round1_text": first[i],
                        "latest_text": latest[i],
                        "final_text": second[i],
                        "from_nocode_retry": from_nocode[i],
                        "from_repair": from_repair[i],
                    })
            out.append(per)
        return out

    pools = generate_pool(rows, args.num_samples, args.seed)

    def verified_counts(per):
        c = Counter()
        for pred, e in per:
            if e is not None and e["status"] == "ok" and pred is not None:
                so = normalize((e.get("stdout") or "").strip())
                if so is not None and so == pred:
                    c[pred] += 1
        return c

    # ── adaptive: plurality 미확정 문제에만 추가 샘플 ──
    if args.adaptive_extra > 0:
        need = []
        for r, per in enumerate(pools):
            c = verified_counts(per)
            top = c.most_common()
            settled = bool(top) and top[0][1] >= args.adaptive_min_count and \
                not (len(top) > 1 and top[0][1] == top[1][1])
            if not settled:
                need.append(r)
        stats["adaptive_targets"] = len(need)
        if need:
            extra = generate_pool([rows[r] for r in need], args.adaptive_extra, args.seed + 555)
            for j, r in enumerate(need):
                pools[r] = pools[r] + extra[j]

    args.output.parent.mkdir(parents=True, exist_ok=True)
    correct = total = adopted = 0
    with args.output.open("w", encoding="utf-8") as f:
        for r, row in enumerate(rows):
            per = pools[r]
            vc = verified_counts(per)
            top = vc.most_common()
            pick = top[0][0] if (top and not (len(top) > 1 and top[0][1] == top[1][1])) else None
            gold = normalize(row.get("answer"))
            rec = {
                "id": row["id"],
                "answer": None if gold is None else str(gold),
                "prediction": None if pick is None else str(pick),
                "verified_counts": {str(k): v for k, v in vc.items()},
                "verified_support": sum(vc.values()),
                "sample_predictions": [None if p is None else str(p) for p, _ in per],
                "sample_exec_status": [None if e is None else e["status"] for _, e in per],
                "n_samples": len(per),
            }
            if gold is not None:
                total += 1
                rec["correct"] = pick is not None and pick == gold
                correct += int(rec["correct"])
            if pick is not None:
                adopted += 1
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    if args.dump_trajectories is not None:
        args.dump_trajectories.parent.mkdir(parents=True, exist_ok=True)
        with args.dump_trajectories.open("w", encoding="utf-8") as f:
            for rec in traj_dump:
                f.write(json.dumps(rec, ensure_ascii=False) + "\n")
        stats["dumped_trajectories"] = len(traj_dump)

    stats["total"] = len(rows)
    stats["adopted"] = adopted
    stats["correct"] = correct
    stats["elapsed_sec"] = round(time.time() - t0, 1)
    print(json.dumps(dict(stats), ensure_ascii=False))
    print(f"correct={correct}/{total} -> {args.output}")


if __name__ == "__main__":
    main()
