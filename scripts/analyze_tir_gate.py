"""TIR gate120 회귀 분해 — TIR을 전면 적용할지, 조건부로 쓸지 판정한다.

핵심 질문: TIR은 baseline을 대체해야 하나(전면), 아니면 특정 조건에서만 채택해야 하나(게이트)?
  - 전면 적용이 net 음수여도, "코드가 정상 실행된 문제에서만 TIR 채택" 같은 규칙이
    양수면 TIR은 여전히 쓸모가 있다. rigor 프롬프트에는 이런 조건 신호가 없었지만
    TIR에는 있다 — 코드 실행 성공 여부는 공짜로 얻는 신뢰도 신호다.
"""
import csv
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
GATE = ROOT / "data/holdout/holdout464_gate120.csv"
TIR = ROOT / "outputs/tir_gate120_hybrid3145.jsonl"
BASE = ROOT / "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl"


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
    with open(GATE, encoding="utf-8-sig", newline="") as f:
        gold = {r["id"]: norm(r["answer"]) for r in csv.DictReader(f)}

    base = {}
    for line in open(BASE):
        r = json.loads(line)
        if r["id"] in gold:
            base[r["id"]] = norm(r.get("prediction"))

    tir = {}
    for line in open(TIR):
        r = json.loads(line)
        tir[r["id"]] = r
    ids = [i for i in gold if i in tir and i in base]

    b_ok = sum(1 for i in ids if base[i] == gold[i])
    t_ok = sum(1 for i in ids if norm(tir[i]["prediction"]) == gold[i])
    print(f"gate120 대상 {len(ids)}문제")
    print(f"  baseline(hybrid_3145) {b_ok}/{len(ids)}")
    print(f"  TIR 전면 적용          {t_ok}/{len(ids)}   ({t_ok-b_ok:+d})")

    gain = [i for i in ids if base[i] != gold[i] and norm(tir[i]["prediction"]) == gold[i]]
    reg = [i for i in ids if base[i] == gold[i] and norm(tir[i]["prediction"]) != gold[i]]
    print(f"  gain {len(gain)} / regression {len(reg)}")
    print()

    print("=== 실행 상태별 분해 ===")
    print(f"{'상태':12s} {'문제수':>5s} {'TIR정답':>7s} {'base정답':>8s} {'차이':>5s}")
    for status in ("ok", "error", "timeout", None):
        sel = [i for i in ids if tir[i].get("exec_status") == status]
        if not sel:
            continue
        t = sum(1 for i in sel if norm(tir[i]["prediction"]) == gold[i])
        b = sum(1 for i in sel if base[i] == gold[i])
        label = {"ok": "실행성공", "error": "실행에러", "timeout": "타임아웃", None: "코드없음"}[status]
        print(f"{label:12s} {len(sel):5d} {t:7d} {b:8d} {t-b:+5d}")
    print()

    print("=== 조건부 채택 규칙 ===")

    def rule(name, accept):
        pred = {i: (norm(tir[i]["prediction"]) if accept(i) else base[i]) for i in ids}
        n = sum(1 for i in ids if pred[i] == gold[i])
        g = sum(1 for i in ids if base[i] != gold[i] and pred[i] == gold[i])
        r = sum(1 for i in ids if base[i] == gold[i] and pred[i] != gold[i])
        used = sum(1 for i in ids if accept(i))
        print(f"{name:44s} {n:3d}/{len(ids)}  ({n-b_ok:+d})  gain {g:2d} reg {r:2d}  TIR채택 {used:3d}")

    rule("baseline 유지 (아무것도 안 함)", lambda i: False)
    rule("TIR 전면 적용", lambda i: True)
    rule("코드 실행 성공한 문제만 TIR", lambda i: tir[i].get("exec_status") == "ok")
    rule("실행 성공 + stdout이 정수인 문제만",
         lambda i: tir[i].get("exec_status") == "ok"
         and norm((tir[i].get("exec_stdout") or "").strip()) is not None)
    rule("실행 성공 + stdout == 최종답인 문제만",
         lambda i: tir[i].get("exec_status") == "ok"
         and norm((tir[i].get("exec_stdout") or "").strip()) is not None
         and norm((tir[i].get("exec_stdout") or "").strip()) == norm(tir[i]["prediction"]))
    rule("실행 성공 + baseline과 답이 같을 때만(무의미·확인용)",
         lambda i: tir[i].get("exec_status") == "ok" and norm(tir[i]["prediction"]) == base[i])
    print()

    agree = [i for i in ids if norm(tir[i]["prediction"]) == base[i]]
    dis = [i for i in ids if norm(tir[i]["prediction"]) != base[i]]
    a_ok = sum(1 for i in agree if base[i] == gold[i])
    print("=== TIR vs baseline 합의 구조 ===")
    print(f"  합의 {len(agree)}문제 -> {a_ok} 정답 ({a_ok/max(len(agree),1):.1%})")
    print(f"  불일치 {len(dis)}문제 -> baseline {sum(1 for i in dis if base[i]==gold[i])}, "
          f"TIR {sum(1 for i in dis if norm(tir[i]['prediction'])==gold[i])}, "
          f"둘 중 하나라도 정답 {sum(1 for i in dis if gold[i] in (base[i], norm(tir[i]['prediction'])))}")


if __name__ == "__main__":
    main()
