"""verifier 를 **tie-break 전용**으로만 쓰는 마지막 변형.

── 왜 이 변형인가 ──────────────────────────────────────────────
재순위(reranked) 판정: 챔피언 대비 -1. 발동 14 -> 57 로 늘면서 gain 4->7,
regression 1->8. 즉 verifier 는 판별력은 있으나(dev 73.8%) **표수 신호를
대체할 만큼은 아니다.** 표를 세는 쪽이 더 믿을 만했다.

그렇다면 대체하지 말고, **현행 규칙이 아예 발동하지 못하는 자리**에만 넣는다:
  - 코드검증 표가 min_count 미만이거나
  - 최다 득표가 동률이라 현행이 기권하는 문제
이 자리에서는 어차피 챔피언 답이 그대로 나가므로, verifier 가 틀려도
잃는 건 '원래도 못 맞히던 문제'뿐이고 regression 상한이 구조적으로 낮다.

현행이 발동한 문제는 **한 글자도 건드리지 않는다** — 그래서 reranked 의
regression 8 이 원천 차단된다.

3가지 임계값을 함께 스윕한다: P(A) 가 낮으면 verifier 자신도 확신이 없다는 뜻이라
그냥 기권하는 편이 낫다.
"""
import argparse
import hashlib
import json
import sys
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "scripts"))
from tir_common import normalize  # noqa: E402


def half(pid):
    return "calib" if int(hashlib.sha256(f"split:{pid}".encode()).hexdigest()[:8], 16) % 2 == 0 else "valid"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--scored", type=Path, required=True,
                    help="apply_verifier_holdout 가 남긴 후보별 P(A) (jsonl)")
    ap.add_argument("--questions", type=Path,
                    default=ROOT / "data/holdout/holdout464_vote3.csv")
    ap.add_argument("--champion", type=Path,
                    default=ROOT / "outputs/champion_holdout464_equivalent.jsonl")
    ap.add_argument("--min-count", type=int, default=2)
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()

    import csv
    gold = {r["id"]: normalize(r.get("answer")) for r in csv.DictReader(
        open(args.questions, encoding="utf-8-sig"))}
    champ = {}
    for line in open(args.champion, encoding="utf-8"):
        r = json.loads(line)
        champ[r["id"]] = normalize(r.get("prediction"))

    cands = defaultdict(list)
    for line in open(args.scored, encoding="utf-8"):
        r = json.loads(line)
        # 덤프의 pred 는 문자열이다. gold/champion 은 normalize 된 정수라 그대로 비교하면
        # 항상 불일치한다 (실제로 이 버그로 판정이 통째로 틀렸다).
        r["pred"] = normalize(r["pred"])
        if r["pred"] is None:
            continue
        cands[r["id"]].append(r)

    def deployed(cs):
        c = Counter(x["pred"] for x in cs if x["code_verified"])
        t = c.most_common()
        if not t or t[0][1] < args.min_count:
            return None
        if len(t) > 1 and t[0][1] == t[1][1]:
            return None
        return t[0][0]

    rows = []
    for thr in (0.0, 0.5, 0.7, 0.8, 0.9, 0.95):
        st = Counter()
        for pid, cs in cands.items():
            g, base = gold.get(pid), champ.get(pid)
            if g is None or base is None:
                continue
            st["n"] += 1
            cur = deployed(cs)
            if cur is not None:
                final = cur                      # 현행이 발동하면 그대로 둔다
            else:
                cv = [x for x in cs if x["code_verified"]]
                best = max(cv, key=lambda x: x["p"]) if cv else None
                if best is not None and best["p"] >= thr:
                    final = best["pred"]
                    if final != base:
                        st["tb_fired"] += 1
                        st["tb_gain"] += int(final == g and base != g)
                        st["tb_reg"] += int(final != g and base == g)
                else:
                    final = base
            st["base"] += int(base == g)
            st["new"] += int(final == g)
            st[f"{half(pid)}_base"] += int(base == g)
            st[f"{half(pid)}_new"] += int(final == g)
        rows.append({
            "threshold": thr, "n": st["n"],
            "champion": st["base"], "rule": st["new"],
            "delta": st["new"] - st["base"],
            "tiebreak_fired": st["tb_fired"], "gain": st["tb_gain"],
            "regression": st["tb_reg"],
            "calib_delta": st["calib_new"] - st["calib_base"],
            "valid_delta": st["valid_new"] - st["valid_base"],
        })

    print(f"{'thr':>5} {'챔피언':>7} {'규칙':>6} {'델타':>6} {'발동':>5} "
          f"{'gain':>5} {'reg':>4} {'calib':>6} {'valid':>6}")
    for r in rows:
        print(f"{r['threshold']:>5.2f} {r['champion']:>7} {r['rule']:>6} {r['delta']:>+6} "
              f"{r['tiebreak_fired']:>5} {r['gain']:>5} {r['regression']:>4} "
              f"{r['calib_delta']:>+6} {r['valid_delta']:>+6}")
    print("\n⚠️ 87문제 표준오차 ±3.5~3.9 (C-3). 델타 4 미만은 실효과로 읽지 말 것.")
    print("   현행 규칙 단독은 +3 이었다 — 이걸 못 넘으면 verifier 는 기여 0이다.")
    args.output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n")


if __name__ == "__main__":
    main()
