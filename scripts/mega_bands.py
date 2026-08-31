"""메가 홀드아웃 밴드 산출 — SC 표수, 5-voter support, risky → 하위 CSV 3종."""
import csv
import json
import re
import sys
from collections import Counter
from pathlib import Path

ROOT = Path("/workspace/DLC")
sys.path.insert(0, str(ROOT / "scripts"))
from extractor_v2 import extract_v2, norm  # noqa: E402

import os
O = Path(os.environ.get("MEGA_OUT", str(ROOT / "outputs/mega")))
IN = Path(os.environ.get("MEGA_IN", str(ROOT / "data/holdout/mega_holdout_2000.csv")))
if not O.is_absolute():
    O = ROOT / O
if not IN.is_absolute():
    IN = ROOT / IN

INT_TAIL = re.compile(r"(?:final answer\s*(?:is|:|=)?\s*\$?\\?\(?\s*-?\d|\\boxed\s*\{\s*-?\d)", re.I)


def risky(text):
    return not INT_TAIL.search((text or "")[-400:])


def jl(p):
    return [json.loads(l) for l in open(p) if l.strip()]


rows = list(csv.DictReader(open(IN, encoding="utf-8-sig")))
q = {r["id"]: r for r in rows}
ids = [r["id"] for r in rows]

voters = {}
for name in ["hybrid3145", "h3244", "ext3000", "h4145", "verify"]:
    d = {}
    for r in jl(O / f"voter_{name}.jsonl"):
        p = r.get("prediction")
        d[r["id"]] = norm(p) if p is not None else norm(extract_v2(r.get("response")))
    voters[name] = d

sc = {r["id"]: r for r in jl(O / "sc_hybrid_n8.jsonl")}

support, votes, nrisky = {}, {}, {}
for i in ids:
    c = Counter(v for v in (voters[n].get(i) for n in voters) if v is not None)
    support[i] = c.most_common(1)[0][1] if c else 0
    r = sc.get(i, {})
    preds = [norm(extract_v2(t)) for t in (r.get("responses") or [])]
    cc = Counter(p for p in preds if p is not None)
    votes[i] = cc.most_common(1)[0][1] if cc else 0
    nrisky[i] = sum(risky(t) for t in (r.get("responses") or []))


def write(path, subset):
    with open(path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["id", "question", "answer"])
        w.writeheader()
        for i in subset:
            w.writerow({"id": i, "question": q[i]["question"],
                        "answer": q[i].get("answer", "")})  # 최종 테스트엔 answer 없음


v3 = [i for i in ids if votes[i] <= 3]
v45r = [i for i in ids if votes[i] in (4, 5) and nrisky[i] >= 1]
s4 = [i for i in ids if support[i] <= 4]
write(O / "mega_vote3.csv", v3)
write(O / "mega_vote45_risky.csv", v45r)
write(O / "mega_sup_le4.csv", s4)
json.dump({"vote_le3": len(v3), "vote45_risky": len(v45r), "support_le4": len(s4),
           "support": support, "votes": votes, "nrisky": nrisky},
          open(O / "mega_bands.json", "w"))
print(f"표수<=3: {len(v3)} | 표수4~5&risky: {len(v45r)} | support<=4: {len(s4)}")
