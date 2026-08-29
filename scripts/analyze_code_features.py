"""코드 내용 기반 선택 신호 — 표 세기가 한계에 부딪힌 뒤의 마지막 통계적 카드.

문제: 87문제에서 코드검증 오라클 48, 실채택 30. 놓친 22개 중 18개가 표 차이 <=2로
아깝게 진다. 표를 어떻게 세도(계보 확대, tie-break, min-count) 뒤집히지 않았다.

가설: 지금 표 세기는 **같은 버그를 4번 반복한 4표**와 **서로 다른 접근 4개가 일치한 4표**를
구분하지 못한다. 후자가 훨씬 강한 증거다.

시험하는 신호:
  distinct   : 그 답을 만든 서로 다른 코드의 개수 (정규화 후 중복 제거)
  sympy      : sympy를 쓴 샘플 비율
  brute_cap  : range() 상한이 작은 임의값인 코드 비율(위험 신호)
  code_len   : 코드 길이 중앙값
  clean_out  : stdout이 정수 한 줄인 비율
"""
import csv
import hashlib
import json
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tir_inference import normalize  # noqa: E402

SUBSET = ROOT / "data/holdout/holdout464_vote3.csv"
BASE = ROOT / "outputs/holdout464_hybrid_3145_baseline_predictions.jsonl"
LINEAGES = {
    "hybrid3145": "outputs/tirc_hybrid3145_holdout464_vote3.jsonl",
    "grpo96": "outputs/tirc_grpo96_holdout464_vote3.jsonl",
    "tirsft": "outputs/tirc_tirsft_holdout464_vote3.jsonl",
}


def canon_code(code):
    """주석·공백·변수명 차이를 지우고 코드의 뼈대만 남긴다."""
    if not code:
        return None
    s = re.sub(r"#.*", "", code)
    s = re.sub(r"\s+", "", s)
    return hashlib.md5(s.encode()).hexdigest()[:12]


def main():
    gold = {r["id"]: normalize(r["answer"])
            for r in csv.DictReader(open(SUBSET, encoding="utf-8-sig"))}
    base = {}
    for line in open(BASE):
        r = json.loads(line)
        if r["id"] in gold:
            base[r["id"]] = normalize(r.get("prediction"))

    # 문제 -> 답 -> 그 답을 낸 샘플들의 특성
    cand = defaultdict(lambda: defaultdict(lambda: {
        "votes": 0, "codes": set(), "sympy": 0, "cap": 0, "lens": [], "clean": 0}))
    loaded = []
    for name, path in LINEAGES.items():
        p = ROOT / path
        if not p.exists():
            print(f"[없음] {name} ({path})")
            continue
        loaded.append(name)
        for line in open(p):
            r = json.loads(line)
            if r["id"] not in gold:
                continue
            codes = r.get("sample_codes") or []
            outs = r.get("sample_stdouts") or []
            preds = r.get("sample_predictions") or []
            stats = r.get("sample_exec_status") or []
            for k, pred in enumerate(preds):
                if k >= len(stats) or stats[k] != "ok":
                    continue
                a = normalize(pred)
                out = normalize((outs[k] or "").strip()) if k < len(outs) else None
                if a is None or out is None or out != a:
                    continue  # 코드검증 통과한 것만
                code = codes[k] if k < len(codes) else None
                e = cand[r["id"]][a]
                e["votes"] += 1
                e["codes"].add(canon_code(code))
                if code:
                    e["lens"].append(len(code))
                    if "sympy" in code:
                        e["sympy"] += 1
                    if re.search(r"range\(\s*\d*\s*,?\s*(\d{1,4})\s*\)", code):
                        e["cap"] += 1
                raw = (outs[k] or "").strip() if k < len(outs) else ""
                if raw.count("\n") == 0 and raw:
                    e["clean"] += 1

    if not loaded:
        print("로드된 계보 없음 — 재생성이 아직 안 끝났습니다.")
        return

    ids = [i for i in sorted(gold) if i in cand and i in base]
    b_all = sum(1 for i in ids if base[i] == gold[i])
    calib = [i for i in ids if int(hashlib.sha256(i.encode()).hexdigest()[:8], 16) % 2 == 0]
    valid = [i for i in ids if i not in set(calib)]
    print(f"계보 {loaded}, 대상 {len(ids)}문제, baseline {b_all}\n")

    # 신호 진단: 정답 후보 vs 오답 후보의 특성 차이
    print("=== 신호 진단 (코드검증 통과 후보들) ===")
    agg = {"정답": Counter(), "오답": Counter()}
    n = {"정답": 0, "오답": 0}
    for i in ids:
        for a, e in cand[i].items():
            key = "정답" if a == gold[i] else "오답"
            n[key] += 1
            agg[key]["votes"] += e["votes"]
            agg[key]["distinct"] += len(e["codes"])
            agg[key]["sympy"] += e["sympy"]
            agg[key]["cap"] += e["cap"]
            agg[key]["clean"] += e["clean"]
            agg[key]["len"] += (sum(e["lens"]) / len(e["lens"])) if e["lens"] else 0
    print(f"{'':10s} {'후보수':>6s} {'평균표':>7s} {'평균 고유코드':>12s} {'sympy비율':>9s} {'작은range':>9s} {'깔끔출력':>8s} {'평균길이':>8s}")
    for key in ("정답", "오답"):
        c = max(n[key], 1)
        v = agg[key]["votes"] / c
        print(f"{key:10s} {n[key]:6d} {v:7.2f} {agg[key]['distinct']/c:12.2f} "
              f"{agg[key]['sympy']/max(agg[key]['votes'],1):9.0%} {agg[key]['cap']/max(agg[key]['votes'],1):9.0%} "
              f"{agg[key]['clean']/max(agg[key]['votes'],1):8.0%} {agg[key]['len']/c:8.0f}")
    print()

    # 선택 규칙: 점수 = votes + w * distinct
    print("=== 선택 규칙: score = votes + w x (고유 코드 수) ===")
    print(f"{'w':>5s} {'mc':>3s} {'점수':>8s} {'차이':>5s} {'gain':>5s} {'reg':>4s} {'cal':>5s} {'val':>5s}")
    rows = []
    for w in (0.0, 0.5, 1.0, 2.0, 3.0):
        for mc in (1, 2, 3, 4):
            pred = {}
            for i in ids:
                scored = [(e["votes"] + w * len(e["codes"]), e["votes"], a)
                          for a, e in cand[i].items()]
                scored.sort(reverse=True)
                if not scored or scored[0][1] < mc:
                    pred[i] = base[i]
                elif len(scored) > 1 and scored[0][0] == scored[1][0]:
                    pred[i] = base[i]
                else:
                    pred[i] = scored[0][2]
            n_ok = sum(1 for i in ids if pred[i] == gold[i])
            g = sum(1 for i in ids if base[i] != gold[i] and pred[i] == gold[i])
            rg = sum(1 for i in ids if base[i] == gold[i] and pred[i] != gold[i])
            nc = sum(1 for i in calib if pred[i] == gold[i]) - sum(1 for i in calib if base[i] == gold[i])
            nv = sum(1 for i in valid if pred[i] == gold[i]) - sum(1 for i in valid if base[i] == gold[i])
            print(f"{w:5.1f} {mc:3d} {n_ok:4d}/{len(ids)} {n_ok-b_all:+5d} {g:5d} {rg:4d} {nc:+5d} {nv:+5d}")
            rows.append((n_ok, w, mc, g, rg, nc, nv))
    best = max(rows)
    print()
    print(f"최고: w={best[1]} mc={best[2]} -> {best[0]}/{len(ids)} ({best[0]-b_all:+d}), "
          f"gain {best[3]} reg {best[4]}, calib {best[5]:+d} valid {best[6]:+d}")
    print("참고 — 현재 제출 규칙(단일계보 mc=2, 표만): +13")


if __name__ == "__main__":
    main()
