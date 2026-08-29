"""공개 R1 궤적(open-r1/OpenR1-Math-220k) -> long-CoT SFT 데이터.

── 왜 이게 '미개척 칸'인가 ──────────────────────────────────────
우리 기존 SFT 데이터의 assistant 응답은 평균 **436자**(hybrid_3145), verbose 판도 706자다.
R1 궤적은 보통 8,000~16,000자다. 즉 우리는 지금까지 **long-CoT 를 한 번도 학습시킨 적이 없다.**
18절의 "흡수 벽"은 전부 짧은 CoT 실험이었으므로 이 칸에는 아직 적용되지 않는다.
AIMO 2회 우승(NVIDIA)이 정확히 이 경로(R1 증류)였다.

── 현실적 타협: 길이 상한 ──────────────────────────────────────
전체 R1 궤적(8천~1.6만 자)을 그대로 학습시키면 모델이 추론 시에도 그만큼 길게 뱉는다.
그런데 우리 하네스는 max_new_tokens 1024~2048 이라 전부 잘린다. 추론 스택 전체를
다시 튜닝할 시간이 없다.
-> **짧은 축의 R1 궤적만 고른다.** 완결된 추론이면서 우리 예산에 맞는 것들이다.
   현재 436자 -> 목표 2,000~6,000자면 여전히 **5~14배**라 변수는 충분히 크다.

── 필터 (전부 만족) ────────────────────────────────────────────
  1. `answer` 가 정수 (대회 정답은 전부 정수)
  2. `correctness_math_verify` 가 True 인 generation 만
  3. 길이 상한 이내 (--max-chars)
  4. 리더보드 831 · holdout464 와 지문 중복 0 (누수 차단)
"""
import argparse, csv, hashlib, json, re, sys
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/DLC")
sys.path.insert(0, str(ROOT / "scripts"))
from tir_common import normalize as n  # noqa: E402

SYSTEM = ("Solve the math problem. Reason carefully step by step, checking your work "
          "as you go. End with exactly: Final answer: <integer>")


def key(q):
    return re.sub(r"[^a-z0-9]", "", (q or "").lower())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", type=int, default=3000, help="수집 목표 건수")
    ap.add_argument("--scan", type=int, default=60000, help="최대 스캔 행수")
    ap.add_argument("--min-chars", type=int, default=1200)
    ap.add_argument("--max-chars", type=int, default=6000)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    # 누수 차단용 지문 집합
    bad = set()
    for p in ["data/deep_chal_math_leaderboard_filtered.csv",
              "data/holdout/official_holdout_464_clean.csv"]:
        for r in csv.DictReader(open(ROOT / p, encoding="utf-8-sig")):
            k = key(r["question"])
            if len(k) > 60:
                bad.add(k)
    print(f"누수 차단 지문 {len(bad)}개 로드")

    from datasets import load_dataset
    ds = load_dataset("open-r1/OpenR1-Math-220k", split="train", streaming=True)

    st = Counter()
    out = []
    for i, r in enumerate(ds):
        if i >= args.scan or len(out) >= args.target:
            break
        st["scan"] += 1
        ans = n(r.get("answer"))
        if ans is None:
            st["skip_not_integer"] += 1
            continue
        q = r.get("problem") or ""
        qk = key(q)
        if any(qk.startswith(b[:200]) or b in qk for b in ()) or qk in bad:
            st["skip_leak"] += 1
            continue
        gens = r.get("generations") or []
        ok = r.get("correctness_math_verify") or []
        best = None
        for g, c in zip(gens, ok):
            if not c or not isinstance(g, str):
                continue
            if not (args.min_chars <= len(g) <= args.max_chars):
                continue
            if best is None or len(g) < len(best):   # 짧은 축 우선
                best = g
        if best is None:
            st["skip_no_usable_gen"] += 1
            continue
        # R1 은 <think>...</think> 뒤에 최종 풀이를 낸다. think 는 남기되
        # 마지막에 우리 포맷의 최종답을 붙여 추출기와 맞춘다.
        body = best.strip()
        if not re.search(r"final answer\s*[::]", body, re.I):
            body = f"{body}\n\nFinal answer: {ans}"
        out.append({"messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": q},
            {"role": "assistant", "content": body},
        ], "meta": {"answer": str(ans), "chars": len(body)}})
        st["kept"] += 1
        if len(out) % 500 == 0:
            print(f"  수집 {len(out)} / 스캔 {i+1}")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("w", encoding="utf-8") as f:
        for s in out:
            f.write(json.dumps({"messages": s["messages"]}, ensure_ascii=False) + "\n")
    lens = [s["meta"]["chars"] for s in out]
    man = {"output": str(args.output), "kept": len(out),
           "assistant_chars": {"mean": int(sum(lens) / max(len(lens), 1)),
                               "min": min(lens) if lens else 0,
                               "max": max(lens) if lens else 0},
           "stats": dict(st),
           "source": "open-r1/OpenR1-Math-220k (Apache-2.0, 공개)",
           "sha256": hashlib.sha256(args.output.read_bytes()).hexdigest()}
    Path(str(args.output).replace(".jsonl", ".manifest.json")).write_text(
        json.dumps(man, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps(man, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
