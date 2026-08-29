"""Report style-absorption indicators for a prediction file.

학습 데이터의 상세 CoT는 전부 `**Approach:**`로 시작하고 평균 1,473자다.
모델 출력이 그쪽으로 움직였는지가 흡수의 직접 지표다.
"""
import json
import sys


def main(path: str, name: str) -> None:
    rows = [json.loads(l) for l in open(path)]
    n = len(rows)
    resp = [r.get("response") or "" for r in rows]
    lens = sorted(len(x) for x in resp)
    correct = sum(1 for r in rows if r.get("correct"))
    approach = sum(1 for x in resp if "**Approach:**" in x)
    header = sum(1 for x in resp if "###" in x)
    verify = sum(1 for x in resp if any(w in x.lower() for w in ("verify", "check", "substitut")))
    trunc = sum(1 for r in rows if r.get("retried_truncated"))
    print(
        f"ABSORB {name}: {correct}/{n} | **Approach:** {approach}/{n} | ### {header}/{n} "
        f"| verify {verify}/{n} | len avg {sum(lens)//n} med {lens[n//2]} | trunc {trunc}"
    )


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2])
