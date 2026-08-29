"""Step 2 채점 — 불일치 51문제에서 cross-policy self-consistency가 선택을 개선하는가.

비교 대상 (전부 같은 51문제):
  - GRPO96 deterministic          = 11/51   (기준선)
  - 기존 16샘플(계보당 8) plurality = 17/51  (현재 최선, 추가 추론 0으로 얻은 값)
  - 신규 포함 32샘플(계보당 16) 각종 규칙  ← 이번에 재는 것

성공/중단 기준 (사전 고정):
  성공 = 21/51 이상, gain >= 2 x regression, calibration/validation 양쪽 양수
  중단 = 19/51 이하 -> 샘플 양으로 선택 병목은 풀리지 않는다

전체 holdout464 환산은 합의 413문제의 349정답에 이 값을 더하면 된다.
"""
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
SUBSET = ROOT / "data/holdout/holdout464_gv_disagree51.csv"
AGREE_CORRECT = 349  # 합의 413문제 중 정답 수 (고정)

DET = {
    "grpo96": "outputs/grpo_3145_passrate94_steps96_holdout464_retry2048.jsonl",
    "verbose": "outputs/verbose_distill_holdout464_retry2048.jsonl",
}
OLD_POOL = {
    "grpo96": "outputs/self_consistency_grpo96_n8_holdout464_seed20260826.jsonl",
    "verbose": "outputs/self_consistency_verbose_n8_holdout464_seed20260827.jsonl",
}
NEW_POOL = {
    "grpo96": "outputs/step2_grpo96_n8_disagree51_seed20260901.jsonl",
    "verbose": "outputs/step2_verbose_n8_disagree51_seed20260902.jsonl",
}


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


def load_det(path):
    out = {}
    for line in open(ROOT / path):
        r = json.loads(line)
        out[r["id"]] = norm(r.get("prediction"))
    return out


def load_pool(path, ids=None):
    p = ROOT / path
    if not p.exists():
        return None
    out = {}
    for line in open(p):
        r = json.loads(line)
        if ids is not None and r["id"] not in ids:
            continue
        out[r["id"]] = [s for s in (norm(x) for x in r.get("sample_predictions", [])) if s is not None]
    return out


def plur(c):
    if not c:
        return None
    r = c.most_common()
    if len(r) > 1 and r[0][1] == r[1][1]:
        return None
    return r[0][0]


def main():
    with open(SUBSET, encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    gold = {r["id"]: norm(r["answer"]) for r in rows}
    ids = sorted(gold)

    det = {k: load_det(v) for k, v in DET.items()}
    old = {k: load_pool(v, set(ids)) for k, v in OLD_POOL.items()}
    new = {k: load_pool(v, set(ids)) for k, v in NEW_POOL.items()}
    missing = [k for k, v in new.items() if v is None]
    if missing:
        print(f"[대기] 아직 생성 안 된 파일: {missing}")
    new = {k: v for k, v in new.items() if v is not None}

    base = {i: det["grpo96"].get(i) for i in ids}
    base_n = sum(1 for i in ids if base[i] == gold[i])

    # calibration / validation 분할 (CONTEXT의 규칙 채택 절차)
    calib = [i for i in ids if int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) % 2 == 0]
    valid = [i for i in ids if i not in set(calib)]

    def counts(i, keys, pools):
        c = Counter()
        for k in keys:
            for p in pools:
                if k in p and i in p[k]:
                    c.update(p[k][i])
        return c

    def evaluate(name, fn):
        pred = {i: fn(i) for i in ids}
        n = sum(1 for i in ids if pred[i] == gold[i])
        gain = sum(1 for i in ids if base[i] != gold[i] and pred[i] == gold[i])
        reg = sum(1 for i in ids if base[i] == gold[i] and pred[i] != gold[i])
        nc = sum(1 for i in calib if pred[i] == gold[i]) - sum(1 for i in calib if base[i] == gold[i])
        nv = sum(1 for i in valid if pred[i] == gold[i]) - sum(1 for i in valid if base[i] == gold[i])
        print(f"{name:44s} {n:2d}/51  464환산 {AGREE_CORRECT+n:3d}  gain {gain:2d} reg {reg:2d}  calib {nc:+2d} valid {nv:+2d}")
        return n

    print(f"51문제 / calibration {len(calib)} · validation {len(valid)}")
    print(f"{'규칙':44s} {'점수':>5s}  {'464환산':>8s}  {'gain/reg':>10s}  {'분할':>12s}")
    evaluate("GRPO96 deterministic (기준선)", lambda i: base[i])

    print("--- 기존 16샘플만 (추가 추론 0) ---")
    for mc in (0, 4, 5, 6, 7, 8):
        def f(i, mc=mc):
            c = counts(i, ["grpo96", "verbose"], [old])
            p = plur(c)
            return p if (p is not None and c[p] >= mc) else base[i]
        evaluate(f"기존16 plurality min{mc}", f)

    if len(new) == 2:
        print("--- 신규 포함 32샘플 (계보당 16) ---")
        for mc in (0, 6, 8, 10, 12, 14, 16):
            def f(i, mc=mc):
                c = counts(i, ["grpo96", "verbose"], [old, new])
                p = plur(c)
                return p if (p is not None and c[p] >= mc) else base[i]
            evaluate(f"32샘플 plurality min{mc}", f)

        print("--- cross-policy 합의 규칙 (계보별 plurality가 일치할 때만 교체) ---")
        for mc in (0, 4, 6, 8):
            def f(i, mc=mc):
                cg = counts(i, ["grpo96"], [old, new])
                cv = counts(i, ["verbose"], [old, new])
                pg, pv = plur(cg), plur(cv)
                if pg is not None and pg == pv and (cg[pg] + cv[pv]) >= mc:
                    return pg
                return base[i]
            evaluate(f"계보별 plurality 일치 + 합산 min{mc}", f)

        print("--- 참고: 오라클 ---")
        orc16 = sum(1 for i in ids if gold[i] in set(counts(i, ["grpo96", "verbose"], [old])))
        orc32 = sum(1 for i in ids if gold[i] in set(counts(i, ["grpo96", "verbose"], [old, new])))
        print(f"  기존16 oracle {orc16}/51 -> 신규포함32 oracle {orc32}/51  (신규 샘플이 연 문제 {orc32-orc16}개)")


if __name__ == "__main__":
    main()
