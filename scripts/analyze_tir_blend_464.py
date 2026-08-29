"""TIR 블렌딩 규칙을 holdout464 전체에서 판정한다.

규칙: (기존 N=8 최다득표 <= vote_threshold) AND (코드 실행 성공 + stdout 정수 + stdout == 최종답)
      -> TIR 답 채택, 그 외 baseline 유지.

두 게이트의 근거:
  - 난이도 게이트: TIR이 실제로 회수한 9문제의 N=8 최다득표가 전부 <=3이었다.
  - 실행 게이트: 이게 없으면 전면 적용 -12, 있으면 +1 (gate120 실측).

CONTEXT의 규칙 채택 절차대로 calibration/validation 분할(id sha256 홀짝)을 통과해야 채택한다.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GOLD = ROOT / "data/holdout/official_holdout_464_clean.csv"
BASE = ROOT / "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl"
POOL = ROOT / "outputs/self_consistency_hybrid3145_n8_holdout500.jsonl"
TIR_FILES = [
    "outputs/tir_gate120_hybrid3145.jsonl",
    "outputs/tir_unsolved68_hybrid3145.jsonl",
    "outputs/tir_remaining289_hybrid3145.jsonl",
]


def norm(v):
    if v is None:
        return None
    s = str(v).strip()
    if not s or s.lower() == "none":
        return None
    try:
        return int(s)
    except ValueError:
        try:
            return int(float(s))
        except ValueError:
            return None


def main():
    with open(GOLD, encoding="utf-8-sig", newline="") as f:
        gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(f)}

    base = {}
    for line in open(BASE):
        r = json.loads(line)
        base[r["id"]] = norm(r.get("prediction"))

    pool = {}
    for line in open(POOL):
        r = json.loads(line)
        pool[r["id"]] = [x for x in (norm(y) for y in r.get("sample_predictions", [])) if x is not None]

    tir = {}
    for path in TIR_FILES:
        p = ROOT / path
        if not p.exists():
            print(f"[없음] {path}")
            continue
        for line in open(p):
            r = json.loads(line)
            tir[r["id"]] = r

    ids = [i for i in sorted(gold) if i in tir and i in base]
    print(f"holdout464 중 TIR 예측 보유: {len(ids)}/464")
    missing_pool = [i for i in ids if i not in pool]
    if missing_pool:
        print(f"  (N=8 풀 없는 문제 {len(missing_pool)}개는 표수 0으로 간주)")
    print()

    def votes(i):
        c = Counter(pool.get(i, []))
        return c.most_common(1)[0][1] if c else 0

    def exec_gate(i):
        r = tir[i]
        if r.get("exec_status") != "ok":
            return False
        so = norm((r.get("exec_stdout") or "").strip())
        return so is not None and so == norm(r["prediction"])

    b_all = sum(1 for i in ids if base[i] == gold[i])
    t_all = sum(1 for i in ids if norm(tir[i]["prediction"]) == gold[i])
    print(f"baseline(hybrid_3145)  {b_all}/{len(ids)}")
    print(f"TIR 전면 적용           {t_all}/{len(ids)}  ({t_all - b_all:+d})")
    print()

    calib = [i for i in ids if int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) % 2 == 0]
    valid = [i for i in ids if i not in set(calib)]

    print(f"=== 블렌딩 규칙 (calibration {len(calib)} / validation {len(valid)}) ===")
    print(f"{'규칙':34s} {'전체':>9s} {'차이':>5s} {'gain':>5s} {'reg':>4s} {'발동':>5s} {'calib':>6s} {'valid':>6s}")
    results = []
    for vt in (1, 2, 3, 4, 5, 6, 8):
        for use_gate in (True, False):
            pred, fired = {}, 0
            for i in ids:
                ok = votes(i) <= vt and (exec_gate(i) if use_gate else True)
                fired += ok
                pred[i] = norm(tir[i]["prediction"]) if ok else base[i]
            n = sum(1 for i in ids if pred[i] == gold[i])
            g = sum(1 for i in ids if base[i] != gold[i] and pred[i] == gold[i])
            rg = sum(1 for i in ids if base[i] == gold[i] and pred[i] != gold[i])
            nc = sum(1 for i in calib if pred[i] == gold[i]) - sum(1 for i in calib if base[i] == gold[i])
            nv = sum(1 for i in valid if pred[i] == gold[i]) - sum(1 for i in valid if base[i] == gold[i])
            label = f"표수<={vt} " + ("+ 실행게이트" if use_gate else "(게이트 없음)")
            print(f"{label:34s} {n:4d}/{len(ids)} {n-b_all:+5d} {g:5d} {rg:4d} {fired:5d} {nc:+6d} {nv:+6d}")
            if use_gate:
                results.append((n, vt, g, rg, nc, nv))
    print()

    best = max(results)
    print(f"최고 규칙: 표수<={best[1]} + 실행게이트 -> {best[0]}/{len(ids)} ({best[0]-b_all:+d}), "
          f"gain {best[2]} reg {best[3]}, calib {best[4]:+d} valid {best[5]:+d}")
    print()
    print("채택 조건 점검:")
    print(f"  순이득 양수            {'통과' if best[0] > b_all else '실패'}")
    print(f"  gain >= 2 x regression {'통과' if best[2] >= 2 * best[3] else '실패'} ({best[2]} vs {best[3]})")
    print(f"  calibration 비음수      {'통과' if best[4] >= 0 else '실패'}")
    print(f"  validation 비음수       {'통과' if best[5] >= 0 else '실패'}")

    print()
    print("=== 참고: TIR이 여는 신규 오라클 ===")
    new = [i for i in ids if base[i] != gold[i] and norm(tir[i]["prediction"]) == gold[i]]
    new_gated = [i for i in new if exec_gate(i)]
    print(f"  baseline 오답인데 TIR이 맞힌 문제: {len(new)}개 (그 중 실행게이트 통과 {len(new_gated)}개)")


if __name__ == "__main__":
    main()
