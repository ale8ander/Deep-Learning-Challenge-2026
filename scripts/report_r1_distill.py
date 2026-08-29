"""R1 long-CoT 증류 판정 — **1차 지표는 응답 길이**다.

기존 SFT 판정은 코드 생성률을 봤지만 여기서는 다르다. 이번에 바꾼 유일한 변수가
'응답 길이'(436자 -> 4,554자 데이터)이므로, 길이가 안 늘면 학습이 안 된 것이다.

  길이 안 늘음            -> 흡수 0. LR 을 올려 재시도
  길이 늘고 점수 하락      -> 22절 패턴(더 장황하지만 더 못 품). 종료
  길이 늘고 점수 유지·상승 -> 성공. GRPO 를 이 위에 얹는 다음 단계로
"""
import json, re, sys, statistics
from pathlib import Path
sys.path.insert(0, "/workspace/DLC/scripts")
from tir_common import normalize as n

R = Path("/workspace/DLC")


def load(model):
    p = R / f"outputs/r1distill_gen_{model}_holdout87.jsonl"
    if not p.exists():
        return None
    return [json.loads(l) for l in open(p) if l.strip()]


def stats(rows):
    lens = [len(r.get("response") or "") for r in rows]
    think = sum(1 for r in rows if "<think>" in (r.get("response") or ""))
    correct = sum(1 for r in rows
                  if n(r.get("prediction")) is not None
                  and n(r.get("prediction")) == n(r.get("answer")))
    trunc = sum(1 for r in rows if r.get("retried_truncated"))
    return {"n": len(rows), "correct": correct,
            "mean_chars": int(statistics.mean(lens)) if lens else 0,
            "median_chars": int(statistics.median(lens)) if lens else 0,
            "think_rate": round(100 * think / max(len(rows), 1), 1),
            "truncated": trunc}


def main():
    new, base = load("r1distill"), load("hybrid3145")
    if new is None or base is None:
        print("평가 산출물이 없다"); return
    a, b = stats(base), stats(new)
    print("=" * 60)
    print("1단계 — 흡수 지표 (이번에 바꾼 변수 = 응답 길이)")
    print("=" * 60)
    print(f"{'지표':<22}{'기준(3145)':>14}{'R1증류':>12}{'배율':>10}")
    ratio = b["mean_chars"] / max(a["mean_chars"], 1)
    print(f"{'평균 응답 길이(자)':<22}{a['mean_chars']:>13}{b['mean_chars']:>12}{ratio:>9.1f}x")
    print(f"{'중앙 응답 길이(자)':<22}{a['median_chars']:>13}{b['median_chars']:>12}")
    print(f"{'<think> 출현율 %':<22}{a['think_rate']:>13}{b['think_rate']:>12}")
    print(f"{'절단 발생':<22}{a['truncated']:>13}{b['truncated']:>12}")
    print()
    print("=" * 60)
    print("2단계 — 점수 (홀드아웃87, greedy 1회)")
    print("=" * 60)
    print(f"{'정답':<22}{a['correct']:>13}{b['correct']:>12}{b['correct']-a['correct']:>+10}")
    print()
    print("=" * 60); print("판정"); print("=" * 60)
    if ratio < 1.5:
        print("✗ 흡수 0 — 길이가 안 늘었다. LR 1e-5 로도 안 움직였으면 r8 용량 문제일 수 있다.")
        print("  (rank 를 올리는 건 18절에서 -24 를 만든 방향이라 신중해야 한다)")
    elif b["correct"] < a["correct"] - 3:
        print("✗ 22절 패턴 재현 — 더 장황해졌는데 더 못 푼다. long-CoT 트랙 종료.")
    elif b["truncated"] > a["truncated"] * 2:
        print("△ 길이는 늘었으나 절단이 급증 — max_new_tokens 를 올려야 판정 가능.")
        print("  하네스 전체 재튜닝이 필요하므로 남은 시간에는 비현실적이다.")
    else:
        print(f"✓ 흡수 성공({ratio:.1f}x) + 점수 유지/상승. **처음으로 SFT 가 통한 칸이다.**")
        print("  다음 단계: 이 위에 GRPO(RLVR)를 얹는다. 단 TIR 풀 병합 실채택이")
        print("  오르는지를 최종 채택 기준으로 삼을 것(오라클만 오른 전례 7회).")
    print("\n⚠️ 87문제 greedy 1회는 표본이 작다. 통과 시 N=8 + 풀 병합으로 재판정할 것.")


if __name__ == "__main__":
    main()
