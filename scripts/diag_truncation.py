"""라운드1 생성이 토큰 상한에 걸려 코드를 못 쓰는 비율을 실측한다.

CONTEXT 11절에서 같은 진단으로 +8을 얻은 적이 있다("코드 미생성 54개 중 50개가
768토큰에서 잘림"). 지금은 상한이 2048 인데 게이트에서 코드 없는 샘플이 21.8% 다.
그게 지시 무시인지 잘림인지 가른다.

`finish_reason == "length"` 가 잘림의 직접 증거다.
"""
import argparse
import csv
import statistics
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tir_common import TIR_SYSTEM, CODE_BLOCK  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default=str(ROOT / "data/holdout/tir_831_gate282.csv"))
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--num-samples", type=int, default=8)
    ap.add_argument("--max-new-tokens", type=int, default=2048)
    ap.add_argument("--model", default="hybrid3145")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=20260905)
    args = ap.parse_args()

    from openai import OpenAI
    cl = OpenAI(base_url="http://localhost:8000/v1", api_key="EMPTY", timeout=1800)
    rows = list(csv.DictReader(open(args.input, encoding="utf-8-sig")))[: args.limit]

    def go(r):
        resp = cl.chat.completions.create(
            model=args.model,
            messages=[{"role": "system", "content": TIR_SYSTEM},
                      {"role": "user", "content": r["question"]}],
            max_tokens=args.max_new_tokens, n=args.num_samples,
            seed=args.seed, temperature=0.7, top_p=0.95)
        return [(c.finish_reason, c.message.content or "") for c in resp.choices]

    with ThreadPoolExecutor(max_workers=args.workers) as p:
        out = list(p.map(go, rows))
    flat = [x for s in out for x in s]
    n = len(flat)
    trunc = [x for x in flat if x[0] == "length"]
    nocode = [x for x in flat if not CODE_BLOCK.findall(x[1])]
    both = [x for x in flat if x[0] == "length" and not CODE_BLOCK.findall(x[1])]

    print(f"상한 {args.max_new_tokens} / 샘플 {n}개 ({len(rows)}문제 x {args.num_samples})")
    print(f"  상한 도달(잘림)        : {len(trunc):>4} ({len(trunc)/n:.1%})")
    print(f"  코드 블록 없음          : {len(nocode):>4} ({len(nocode)/n:.1%})")
    print(f"  잘려서 코드 없음        : {len(both):>4} ({len(both)/n:.1%})")
    if nocode:
        print(f"  -> 코드 없는 샘플 중 잘린 비율: {len(both)/len(nocode):.1%}")
    withcode = [len(x[1]) for x in flat if CODE_BLOCK.findall(x[1])]
    if withcode:
        print(f"  코드 있는 샘플 평균 길이: {statistics.mean(withcode):.0f}자")
    if nocode:
        print(f"  코드 없는 샘플 평균 길이: {statistics.mean([len(x[1]) for x in nocode]):.0f}자")


if __name__ == "__main__":
    main()
