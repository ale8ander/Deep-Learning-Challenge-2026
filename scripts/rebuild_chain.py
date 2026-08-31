"""챔피언 제출 체인을 원본 jsonl 에서 통째로 재생성한다.

목적은 두 가지다.
  1. **재현 검증**: 현행 추출기(v1)로 재생성한 결과가 실제 제출된 CSV 와 한 글자도
     다르지 않아야 한다. 이게 통과해야만 v2 로 바꿨을 때 생긴 차이를 추출기 탓으로
     돌릴 수 있다. (2026-08-27 감사에서 5-voter 는 이미 불일치 0 으로 확인됐다.)
  2. **v2 적용**: 통과하면 같은 코드로 v2 체인을 만들고 단계별 변경 수를 보고한다.

체인 (각 단계는 앞 단계 위에 얹는다):
    5-voter(majority, tie=3145)
      -> support=4 문항에만 SC N=8 min-count 4 override      = 챔피언 (Public 623)
      -> 표수<=3 AND TIR SC 코드검증 plurality >= 2            = tir_sc8      (Public 643)
      -> 표수 4~5 AND TIR SC 코드검증 plurality >= 4           = tir_sc8+v45  (Public 647)

사용:
    python3 scripts/rebuild_chain.py --extractor v1 --verify-only
    python3 scripts/rebuild_chain.py --extractor v2 --out-prefix submissions/submission_v2ext
"""
import argparse
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extractor_v2 import extract_v1, extract_v2, norm  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent
TEST = ROOT / "data/deep_chal_math_leaderboard_filtered.csv"

# 순서가 곧 voter 순서다. voter1(=hybrid_3145)이 동률 fallback.
VOTERS = [
    ("hybrid_3145", "outputs/hybrid_3145_leaderboard_retry2048.jsonl"),
    ("hybrid_3244", "outputs/hybrid_3244_leaderboard_retry2048.jsonl"),
    ("external_3000", "outputs/external_3000_r8_qv_lr2e6_e1_leaderboard_retry2048.jsonl"),
    ("hybrid_4145", "outputs/hybrid_4145_r8_qv_lr1p5e6_e1_leaderboard_retry2048.jsonl"),
    ("hybrid_3145_verify", "outputs/hybrid_3145_verify_leaderboard_retry2048.jsonl"),
]
SC_FILES = [
    "outputs/self_consistency_hybrid3145_n8_leaderboard_support1to3.jsonl",
    "outputs/self_consistency_hybrid3145_n8_leaderboard_support4.jsonl",
    "outputs/self_consistency_hybrid3145_n8_leaderboard_support5.jsonl",
]
TIR_VOTE3 = "outputs/tir_sc8_831_vote3_to60.jsonl"
TIR_VOTE45 = "outputs/tirc_831_vote45.jsonl"

REFERENCE = {
    "5voter": "submissions/submission_ensemble_5voter_3145_3244_external_4145_verify.csv",
    "champion": "submissions/submission_self_consistency_hybrid3145_n8_min4_support4.csv",
    "tir_sc8": "submissions/submission_tir_sc8_vote3_mc2.csv",
    "tir_sc8_v45": "submissions/submission_tir_sc8_vote3_plus_vote45mc4.csv",
}


def load_csv(path):
    rows = list(csv.DictReader(open(path, encoding="utf-8-sig")))
    key = "answer" if "answer" in rows[0] else list(rows[0].keys())[1]
    return {r["id"]: norm(r[key]) for r in rows}


def read_jsonl(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                yield json.loads(line)


INT_FINAL = re.compile(
    r"(?:final answer|정답)\s*(?:is|:|=)?\s*\$?\\?\(?\s*(-?\d[\d,]*)\s*\\?\)?\$?\s*\.?\s*$",
    re.I | re.M)
BOXED_INT = re.compile(r"\\boxed\s*\{\s*(-?\d[\d,]*)\s*\}")


def risky(text):
    """응답이 '정수 형태의 최종답'을 냈는가. 답이 무엇인지와는 무관한 형태 판정이다.

    표수와 독립적인 신뢰도 신호다 (holdout464, 표수<=3 구간에서 risky 0개면 47%,
    1개 이상이면 14~18%). CONTEXT 21절에서 소진됐다던 신호들은 전부 표를 세는
    것이었는데 이건 축이 다르다.
    """
    if not text:
        return True
    tail = text[-400:]
    return not (INT_FINAL.search(tail) or BOXED_INT.search(tail))


def plurality(values, min_count, strict_unique=True):
    """최다 득표 값. min_count 미만이거나 동률이면 None."""
    counts = Counter(v for v in values if v is not None)
    if not counts:
        return None
    top = counts.most_common()
    if top[0][1] < min_count:
        return None
    if strict_unique and len(top) > 1 and top[0][1] == top[1][1]:
        return None
    return top[0][0]


def build(extract, test_ids, risky_gate=False, risky_bands=("vote3", "vote45")):
    """추출기 하나로 전체 체인을 만든다. 단계별 dict 를 돌려준다.

    risky_gate=True 면 TIR override 규칙을 홀드아웃에서 검증된 새 규칙으로 바꾼다:
    두 구간(표수<=3, 표수 4~5) 모두 `risky>=1 인 문제에만 min-count 1`.
    홀드아웃에서 표수<=3 은 +14(reg 0, 현행 +12), 표수4~5 는 +3(reg 0, 현행 +2).
    """
    stages = {}

    # --- 1) voter 별 예측 (response 재파싱) ---
    voter_preds = []
    for name, rel in VOTERS:
        preds = {}
        for r in read_jsonl(ROOT / rel):
            preds[r["id"]] = norm(extract(r.get("response")))
        voter_preds.append((name, preds))

    # --- 2) 5-voter majority, 동률/무다수 시 voter1 fallback ---
    # ensemble_predictions.create_submission 과 동일 규약:
    # 최다 득표가 유일하고 2표 이상이면 채택, 아니면 fallback voter 의 답.
    five = {}
    support = {}
    for i in test_ids:
        votes = [p.get(i) for _, p in voter_preds]
        counts = Counter(v for v in votes if v is not None)
        support[i] = counts.most_common(1)[0][1] if counts else 0
        winners = [a for a, c in counts.items() if c == max(counts.values())] if counts else []
        if len(winners) == 1 and max(counts.values()) >= 2:
            five[i] = winners[0]
        else:
            five[i] = votes[0]
    stages["5voter"] = five

    # --- 3) SC N=8 풀 (responses 재파싱) ---
    sc_samples = {}
    sc_nrisky = {}
    for rel in SC_FILES:
        for r in read_jsonl(ROOT / rel):
            resp = r.get("responses")
            if resp:
                sc_samples[r["id"]] = [norm(extract(t)) for t in resp]
                sc_nrisky[r["id"]] = sum(risky(t) for t in resp)
            else:  # responses 가 없으면 저장된 추출 결과를 쓴다
                sc_samples[r["id"]] = [norm(v) for v in r.get("sample_predictions", [])]
                sc_nrisky[r["id"]] = 0

    # --- 4) 챔피언: support==4 문항에만 SC min-count 4 override ---
    champion = dict(five)
    champ_override = []
    for i in test_ids:
        if support.get(i) != 4:
            continue
        pick = plurality(sc_samples.get(i, []), min_count=4)
        if pick is not None and pick != champion[i]:
            champion[i] = pick
            champ_override.append(i)
    stages["champion"] = champion

    # --- 5) SC 표수 (난이도 게이트용) ---
    def sc_votes(i):
        c = Counter(v for v in sc_samples.get(i, []) if v is not None)
        return c.most_common(1)[0][1] if c else 0

    def tir_pick(rec, min_count):
        counts = {norm(k): v for k, v in rec["verified_counts"].items()}
        return plurality([k for k, v in counts.items() if k is not None for _ in range(v)],
                         min_count=min_count)

    # --- 6) TIR SC override: 표수<=3 ---
    tir3 = {r["id"]: r for r in read_jsonl(ROOT / TIR_VOTE3)}
    sc8 = dict(champion)
    fired3 = 0
    for i in test_ids:
        if sc_votes(i) > 3 or i not in tir3:
            continue
        if risky_gate and "vote3" in risky_bands:
            if sc_nrisky.get(i, 0) < 1:
                continue
            pick = tir_pick(tir3[i], 1)
        else:
            pick = tir_pick(tir3[i], 2)
        if pick is not None:
            sc8[i] = pick
            fired3 += 1
    stages["tir_sc8"] = sc8

    # --- 7) 게이트 확장: 표수 4~5 ---
    tir45 = {r["id"]: r for r in read_jsonl(ROOT / TIR_VOTE45)}
    sc8v45 = dict(sc8)
    fired45 = 0
    for i in test_ids:
        if not (4 <= sc_votes(i) <= 5) or i not in tir45:
            continue
        if risky_gate and "vote45" in risky_bands:
            if sc_nrisky.get(i, 0) < 1:
                continue
            pick = tir_pick(tir45[i], 1)
        else:
            pick = tir_pick(tir45[i], 4)
        if pick is not None:
            sc8v45[i] = pick
            fired45 += 1
    stages["tir_sc8_v45"] = sc8v45

    return stages, {"support": support, "sc_votes": {i: sc_votes(i) for i in test_ids},
                    "champ_override": champ_override, "fired3": fired3, "fired45": fired45,
                    "nrisky": sc_nrisky}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--extractor", choices=["v1", "v2"], default="v1")
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--out-prefix", default=None)
    ap.add_argument("--risky-gate", action="store_true",
                    help="TIR override 를 risky>=1 + min-count 1 규칙으로 바꾼다")
    ap.add_argument("--risky-bands", default="vote3,vote45",
                    help="risky 규칙을 적용할 구간. vote3=표수<=3, vote45=표수4~5")
    args = ap.parse_args()

    test_ids = [r["id"] for r in csv.DictReader(open(TEST, encoding="utf-8-sig"))]
    extract = extract_v1 if args.extractor == "v1" else extract_v2

    bands = tuple(b.strip() for b in args.risky_bands.split(",") if b.strip())
    stages, meta = build(extract, test_ids, risky_gate=args.risky_gate, risky_bands=bands)

    print(f"=== 추출기 {args.extractor}"
          f"{' / risky 게이트' if args.risky_gate else ''} / 831문제 ===")
    print(f"5-voter support 분포: {dict(sorted(Counter(meta['support'].values()).items()))}")
    print(f"support==4 SC override: {len(meta['champ_override'])}문제")
    print(f"TIR 발동: 표수<=3 {meta['fired3']}문제 / 표수4~5 {meta['fired45']}문제")
    print()

    print(f"{'단계':<14} {'참조 CSV 대비 불일치':>20}")
    for name, ref in REFERENCE.items():
        # 참조 CSV 는 저장소에 없다 (제출본은 주최측 직접 전달). 로컬에 있을 때만 대조하고,
        # 없으면 reproduce_all.sh compose 가 --out-prefix 출력의 sha256 으로 검증한다.
        if not (ROOT / ref).exists():
            print(f"{name:<14} {'참조 없음(생략)':>20}   ({ref})")
            continue
        refmap = load_csv(ROOT / ref)
        diff = [i for i in test_ids if stages[name][i] != refmap[i]]
        mark = "일치" if not diff else f"{len(diff)}개 불일치"
        print(f"{name:<14} {mark:>20}   ({ref})")
        if diff and len(diff) <= 10:
            print(f"               예: {diff[:10]}")

    if args.verify_only:
        return

    if args.out_prefix:
        for name in REFERENCE:
            out = ROOT / f"{args.out_prefix}_{name}.csv"
            with out.open("w", encoding="utf-8", newline="") as f:
                w = csv.writer(f)
                w.writerow(["id", "answer"])
                for i in test_ids:
                    w.writerow([i, stages[name][i]])
            assert all(stages[name][i] is not None for i in test_ids)
            print(f"저장: {out}")


if __name__ == "__main__":
    main()
