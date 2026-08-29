"""TIR 자기증류 continuation 의 채택 게이트.

CONTEXT 17절의 판정 순서를 코드로 고정한다: **점수보다 흡수 지표를 먼저 본다.**
코드 생성률/실행 성공률/코드검증 오라클이 안 움직이면 학습이 아무것도 안 한 것이고,
그 상태의 점수 차이는 노이즈다. 22절의 두 실패도 흡수는 됐는데 품질이 떨어진 경우라
두 축을 같이 봐야 판별된다.

| 실패 유형 | 흡수 지표 | 오라클 |
|---|---|---|
| 아무것도 안 배움 (19절, 마스킹 버그) | 변화 없음 | 변화 없음 |
| 배웠지만 더 못 씀 (22절) | 코드 생성률↑ | **↓** |
| 성공 | 코드 생성률↑ 또는 유지 | **↑** |

⚠️ 홀드아웃87 은 gain/reg 표준오차가 ±3.5~3.9다 (CONTEXT C-3). 이 스크립트가 찍는
±4 미만 차이는 실효과로 읽지 말 것 — 그래서 오라클과 흡수 지표를 같이 본다.
"""
import argparse
import json
import re
from pathlib import Path


def load(path):
    return [json.loads(l) for l in open(path, encoding="utf-8") if l.strip()]


def load_stats(path):
    """클라이언트가 마지막 줄 앞에 찍는 통계 JSON 을 회수한다."""
    for line in reversed(Path(path).read_text(encoding="utf-8").splitlines()):
        line = line.strip()
        if line.startswith("{") and '"r1_no_code"' in line:
            return json.loads(line)
    return {}


def oracle(rows):
    """샘플 중 하나라도 정답이면 오라클 정답 — 후보 생성 능력의 상한."""
    n = 0
    for r in rows:
        gold = r.get("answer")
        if gold is None:
            continue
        if any(p is not None and str(p) == str(gold) for p in r["sample_predictions"]):
            n += 1
    return n


def code_verified_oracle(rows):
    """코드검증(verified_counts)에 정답이 들어 있는 문제 수. 실제 채택 규칙이 볼 수 있는 상한."""
    n = 0
    for r in rows:
        gold = r.get("answer")
        if gold is None:
            continue
        if str(gold) in (r.get("verified_counts") or {}):
            n += 1
    return n


def adopted_correct(rows):
    return sum(1 for r in rows if r.get("correct"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--new", required=True)
    ap.add_argument("--base", required=True)
    ap.add_argument("--new-log", required=True)
    ap.add_argument("--base-log", required=True)
    args = ap.parse_args()

    new, base = load(args.new), load(args.base)
    ns, bs = load_stats(args.new_log), load_stats(args.base_log)

    def rate(s, key):
        tot = s.get("r1_no_code", 0) + s.get("r1_ok", 0) + s.get("r1_error", 0) + s.get("r1_timeout", 0)
        return 100.0 * s.get(key, 0) / tot if tot else 0.0

    print("=" * 66)
    print("1단계 — 흡수 지표 (1라운드 원본 행동. 리페어/재시도 적용 전)")
    print("=" * 66)
    print(f"{'지표':<24}{'기준(3145)':>14}{'신규':>12}{'차이':>12}")
    rows = [
        ("코드 미생성률 %", rate(bs, "r1_no_code"), rate(ns, "r1_no_code"), -1),
        ("실행 성공률 %", rate(bs, "r1_ok"), rate(ns, "r1_ok"), +1),
        ("실행 에러율 %", rate(bs, "r1_error"), rate(ns, "r1_error"), -1),
    ]
    absorbed = False
    for name, b, n, good in rows:
        d = n - b
        mark = "✓" if d * good > 0 else ("–" if abs(d) < 0.5 else "✗")
        print(f"{name:<24}{b:>13.1f}{n:>12.1f}{d:>+11.1f}  {mark}")
        if abs(d) >= 2.0:
            absorbed = True

    print()
    print("=" * 66)
    print("2단계 — 후보 품질 (87문제)")
    print("=" * 66)
    b_or, n_or = oracle(base), oracle(new)
    b_cv, n_cv = code_verified_oracle(base), code_verified_oracle(new)
    b_ad, n_ad = adopted_correct(base), adopted_correct(new)
    print(f"{'지표':<24}{'기준(3145)':>14}{'신규':>12}{'차이':>12}")
    print(f"{'샘플 오라클':<24}{b_or:>13}{n_or:>12}{n_or - b_or:>+11}")
    print(f"{'코드검증 오라클':<24}{b_cv:>13}{n_cv:>12}{n_cv - b_cv:>+11}")
    print(f"{'실제 채택 정답':<24}{b_ad:>13}{n_ad:>12}{n_ad - b_ad:>+11}")

    print()
    print("=" * 66)
    print("판정")
    print("=" * 66)
    d_cv = n_cv - b_cv
    if not absorbed:
        print("✗ 중단 — 흡수 지표가 안 움직였다. 19절(마스킹 버그)과 같은 '아무것도 안 배움'")
        print("  상태일 수 있다. LR 을 한 단계 올려 재시도하거나(2e-6 초과 금지) 학습 로그의")
        print("  loss 하강 추세를 먼저 확인할 것.")
    elif d_cv < 0:
        print("✗ 중단 — 흡수는 됐으나 코드검증 오라클이 떨어졌다. 22절의 '더 자주 쓰지만")
        print("  더 못 쓴다' 패턴이 자기증류에서도 재현된 것이다. TIR SFT 트랙을 최종 종료한다.")
    elif d_cv == 0:
        print("– 판정 보류 — 흡수는 됐지만 오라클이 그대로다. 87문제 표준오차(±3.5~3.9) 안이라")
        print("  이 차이로는 아무것도 말할 수 없다. 확대 전 holdout464 전체로 재측정할 것.")
    else:
        print(f"✓ 다음 단계 — 코드검증 오라클 +{d_cv}. 단 이건 '상한'이 올랐다는 뜻일 뿐이다.")
        print("  CONTEXT 21절: 오라클만 오르고 실채택 전환이 0 이었던 전례가 6번 있다.")
        print("  **채택 조건은 기존 A100+NC 풀에 이 모델 풀을 더했을 때 실채택이 오르는 것**이고,")
        print("  그건 holdout464 전체에서 compose_holdout464.py 로 재야 한다.")
    print()
    print("어떤 경우에도 이 결과만으로 Public 제출하지 말 것 (홀드아웃→Public 3연패 전례).")


if __name__ == "__main__":
    main()
