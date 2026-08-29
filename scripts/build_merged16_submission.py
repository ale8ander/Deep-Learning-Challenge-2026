"""현행 최고 제출본(656)을 원본 jsonl 에서 통째로 재생성한다.

CONTEXT.md K절의 규칙을 코드로 고정한 것이다. 이 규칙은 지금까지 어느 스크립트에도
없었고 세션마다 손으로 조립됐다(부채 I-1). 주최측 재현 검증에서 이게 없으면
"어떻게 만들었는지 설명할 수 없는 제출본"이 된다.

규칙 (챔피언 623 위에 얹는다):
    표수<=3        : A100 8샘플 + NC 8샘플 = 16샘플, min-count 3
    표수4~5 & risky>=1 : A100 8샘플 + NC 8샘플 = 16샘플, min-count 2
    나머지          : 챔피언 그대로

`표수`는 SC N=8 풀의 최다득표수, `risky`는 SC 응답 8개 중 마지막 400자에 정수 형태
최종답이 없는 샘플 수다(rebuild_chain.risky). 둘 다 rebuild_chain 에서 그대로 가져온다.

사용:
    python3 scripts/build_merged16_submission.py --verify-only
    python3 scripts/build_merged16_submission.py --out submissions/submission_x.csv
"""
import argparse
import csv
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extractor_v2 import extract_v1, norm  # noqa: E402
import rebuild_chain  # noqa: E402  (VOTERS/SC_FILES 를 갈아끼우려면 모듈로 잡아야 한다)
from rebuild_chain import TEST, build, load_csv, plurality, read_jsonl  # noqa: E402

ROOT = Path(__file__).resolve().parent.parent

# A100 pod 산출물 (표수 구간별로 파일이 다르다)
POOL_A100_VOTE3 = "outputs/tir_sc8_831_vote3_to60.jsonl"
POOL_A100_VOTE45 = "outputs/tirc_831_vote45.jsonl"
# 5090 pod 산출물 (게이트 282문제 = 표수<=3 173 + 표수4~5 109 를 한 파일에 담는다)
POOL_NC_GATE282 = "outputs/tir_nocode_831_gate282.jsonl"

REFERENCE = "submissions/submission_nocode_merged16.csv"


def load_counts(rel):
    """id -> Counter(답 -> 코드검증 통과 샘플 수)."""
    out = {}
    for r in read_jsonl(ROOT / rel):
        c = Counter()
        for k, v in (r.get("verified_counts") or {}).items():
            k = norm(k)
            if k is not None:
                c[k] += v
        out[r["id"]] = c
    return out


def merged_pick(pools, pid, min_count):
    """여러 풀의 verified_counts 를 합쳐 plurality 를 고른다.

    풀을 '교체'가 아니라 '합치는' 것이 핵심이다 (CONTEXT B절). 서로 다른 pod/엔진에서
    나왔어도 같은 모델의 독립 추첨이므로 표를 그대로 더한다.
    """
    total = Counter()
    for p in pools:
        total += p.get(pid, Counter())
    if not total:
        return None
    flat = [k for k, v in total.items() for _ in range(v)]
    return plurality(flat, min_count=min_count)


def export_gates(test_ids, sc_votes, outdir):
    """표수 구간별 문제 집합을 CSV 로 떨어뜨린다 (TIR 생성 입력용).

    전체 재생성 경로에서 필요하다 — 게이트는 5-voter 와 SC 표수에서 유도되는 값이라
    앞 단계가 끝나기 전에는 확정할 수 없다.
    """
    src = {r["id"]: r for r in csv.DictReader(open(TEST, encoding="utf-8-sig"))}
    fields = list(next(iter(src.values())).keys())
    bands = {"vote3": lambda v: v <= 3, "vote45": lambda v: 4 <= v <= 5,
             "gate282": lambda v: v <= 5}
    for name, keep in bands.items():
        rows = [src[i] for i in test_ids if keep(sc_votes.get(i, 0))]
        out = Path(outdir) / f"repro_831_{name}.csv"
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.DictWriter(f, fieldnames=fields)
            w.writeheader()
            w.writerows(rows)
        print(f"게이트 저장: {out} ({len(rows)}문제)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--verify-only", action="store_true")
    ap.add_argument("--out", default=None, help="저장할 CSV 경로(미지정 시 저장 안 함)")
    ap.add_argument("--vote3-pools", default=f"{POOL_A100_VOTE3},{POOL_NC_GATE282}",
                    help="표수<=3 구간에서 합칠 TIR 풀 jsonl (쉼표 구분)")
    ap.add_argument("--vote45-pools", default=f"{POOL_A100_VOTE45},{POOL_NC_GATE282}",
                    help="표수4~5 구간에서 합칠 TIR 풀 jsonl (쉼표 구분)")
    ap.add_argument("--reference", default=REFERENCE,
                    help="대조할 기존 제출본. 전체 재생성 경로에서는 샘플이 달라 "
                         "불일치가 정상이므로 --no-reference 로 끈다")
    ap.add_argument("--no-reference", action="store_true")
    ap.add_argument("--vote3-min-count", type=int, default=3,
                    help="표수<=3 구간 min-count. 656 규칙은 3(16샘플). "
                         "24샘플에서는 홀드아웃 실측상 2가 최적(+9 vs +5)")
    ap.add_argument("--vote45-min-count", type=int, default=2,
                    help="표수4~5(risky>=1) 구간 min-count")
    ap.add_argument("--export-gates", default=None,
                    help="표수 구간 CSV 를 이 디렉터리에 저장하고 종료")
    ap.add_argument("--voters", nargs="+", default=None,
                    help="voter jsonl 5개를 직접 지정 (전체 재생성 경로용). 미지정 시 "
                         "rebuild_chain 의 기본 경로(원본 산출물)를 쓴다")
    ap.add_argument("--sc-files", nargs="+", default=None,
                    help="SC tier jsonl 을 직접 지정 (전체 재생성 경로용)")
    args = ap.parse_args()

    # ⚠️ 전체 재생성 경로에서 이걸 안 하면 챔피언·표수·risky 가 **옛 산출물**에서 계산되고
    # TIR 풀만 새 것이 섞인다. 에러 없이 조용히 불일치가 생기는 자리다.
    if args.voters:
        rebuild_chain.VOTERS = [(f"voter{i+1}", p) for i, p in enumerate(args.voters)]
    if args.sc_files:
        rebuild_chain.SC_FILES = list(args.sc_files)

    test_ids = [r["id"] for r in csv.DictReader(open(TEST, encoding="utf-8-sig"))]

    # 챔피언(623)과 표수/risky 메타를 재생성한다.
    stages, meta = build(extract_v1, test_ids)
    champion = stages["champion"]
    sc_votes, nrisky = meta["sc_votes"], meta["nrisky"]

    if args.export_gates:
        export_gates(test_ids, sc_votes, args.export_gates)
        return 0

    v3_pools = [load_counts(p) for p in args.vote3_pools.split(",") if p.strip()]
    v45_pools = [load_counts(p) for p in args.vote45_pools.split(",") if p.strip()]

    final = dict(champion)
    fired3 = fired45 = 0
    for i in test_ids:
        v = sc_votes.get(i, 0)
        if v <= 3:
            pick = merged_pick(v3_pools, i, min_count=args.vote3_min_count)
            if pick is not None:
                final[i] = pick
                fired3 += 1
        elif 4 <= v <= 5 and nrisky.get(i, 0) >= 1:
            pick = merged_pick(v45_pools, i, min_count=args.vote45_min_count)
            if pick is not None:
                final[i] = pick
                fired45 += 1

    print(f"=== 제출본 조립 / {len(test_ids)}문제 "
          f"(표수<=3 mc{args.vote3_min_count}, 표수4~5 mc{args.vote45_min_count}) ===")
    print(f"TIR 발동: 표수<=3 {fired3}문제 / 표수4~5(risky>=1) {fired45}문제")
    print(f"챔피언(623) 대비 답 변경: {sum(final[i] != champion[i] for i in test_ids)}문제")

    diff = []
    if not args.no_reference:
        ref = load_csv(ROOT / args.reference)
        diff = [i for i in test_ids if final[i] != ref[i]]
        if diff:
            print(f"\n[실패] {args.reference} 대비 {len(diff)}개 불일치: {diff[:10]}")
        else:
            print(f"\n[통과] {args.reference} 와 831문제 전부 일치")

    if args.verify_only:
        return 1 if diff else 0

    if args.out:
        assert all(final[i] is not None for i in test_ids), "빈 답이 있다"
        out = ROOT / args.out
        with out.open("w", encoding="utf-8", newline="") as f:
            w = csv.writer(f)
            w.writerow(["id", "answer"])
            for i in test_ids:
                w.writerow([i, final[i]])
        print(f"저장: {out}")
    return 1 if diff else 0


if __name__ == "__main__":
    sys.exit(main())
