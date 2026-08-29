"""TIR SFT 흡수 지표 비교 — 점수보다 먼저 이걸 본다.

오전 세션의 교훈: 학습 강도가 부족하면 "안전한데 아무것도 안 배우고", 과하면 "배우지만 붕괴한다".
그 판정을 점수(노이즈 큼)가 아니라 **행동 지표**로 먼저 한다.

TIR에서 흡수 여부를 보는 지표 3개:
  1. 코드 생성률   — 지시대로 ```python 블록을 쓰는가        (현재 71%)
  2. 실행 성공률   — 그 코드가 실제로 돌아가는가              (현재 61%)
  3. 코드검증 오라클 — 코드가 정답을 만들어내는 문제 수        (현재 33/87)  <- 진짜 관문

3번이 안 오르면 "포맷만 배웠고 알고리즘은 그대로"라는 뜻이고, 그럼 점수도 안 오른다.
"""
import argparse
import json
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tir_inference import normalize  # noqa: E402


def load(path):
    rows = [json.loads(l) for l in open(path)]
    return {r["id"]: r for r in rows}


def stats(rows, label):
    n = len(rows)
    samples = 0
    st = Counter()
    for r in rows.values():
        statuses = r.get("sample_exec_status") or [r.get("exec_status")]
        samples += len(statuses)
        st.update(statuses)
    code = samples - st[None]
    ok = st["ok"]

    # 코드검증 오라클: 코드가 정상 실행돼 나온 답 중 정답이 있는 문제 수
    orc_ver = 0
    orc_any = 0
    for r in rows.values():
        gold = normalize(r.get("answer"))
        if gold is None:
            continue
        vc = r.get("verified_counts")
        if vc:
            verified = {normalize(k) for k in vc}
            allp = {normalize(p) for p in r.get("sample_predictions", [])}
        else:
            verified = {normalize(r.get("prediction"))} if r.get("exec_status") == "ok" else set()
            allp = {normalize(r.get("prediction"))}
        orc_ver += int(gold in verified)
        orc_any += int(gold in allp)

    print(f"{label:32s} 문제 {n:3d}  샘플 {samples:4d}")
    print(f"{'':32s}   코드 생성 {code:4d} ({code/max(samples,1):4.0%})   "
          f"실행 성공 {ok:4d} ({ok/max(samples,1):4.0%})   "
          f"에러 {st['error']:3d}  타임아웃 {st['timeout']:3d}")
    print(f"{'':32s}   코드검증 오라클 {orc_ver:3d}/{n}   전체 오라클 {orc_any:3d}/{n}")
    return {"code_rate": code / max(samples, 1), "ok_rate": ok / max(samples, 1),
            "orc_ver": orc_ver, "orc_any": orc_any, "n": n}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="+", required=True,
                    help="라벨=경로 형식. 예: hybrid3145=outputs/a.jsonl tirsft=outputs/b.jsonl")
    args = ap.parse_args()

    results = {}
    for spec in args.files:
        label, path = spec.split("=", 1)
        p = Path(path)
        if not p.exists():
            print(f"{label}: 파일 없음 ({path})")
            continue
        results[label] = stats(load(p), label)
        print()

    if len(results) >= 2:
        labels = list(results)
        base, new = results[labels[0]], results[labels[-1]]
        print("=== 흡수 판정 ===")
        print(f"  코드 생성률      {base['code_rate']:.0%} -> {new['code_rate']:.0%}  "
              f"({'흡수됨' if new['code_rate'] - base['code_rate'] > 0.10 else '변화 미미'})")
        print(f"  실행 성공률      {base['ok_rate']:.0%} -> {new['ok_rate']:.0%}")
        print(f"  코드검증 오라클  {base['orc_ver']}/{base['n']} -> {new['orc_ver']}/{new['n']}  "
              f"({'후보 생성 개선' if new['orc_ver'] > base['orc_ver'] else '개선 없음 — 포맷만 배움'})")


if __name__ == "__main__":
    main()
