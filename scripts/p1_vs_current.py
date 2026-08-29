"""P1 판정 — C-1 준수판. 기준은 챔피언이 아니라 **현행 배포 규칙**이다.

현행(656) 표수4~5 규칙 = risky>=1 인 문제만, A100+NC1 16샘플 mc2.
홀드아웃 등가물 = A100 + V45n(리페어1+코드미생성재시도1), mc2, risky 게이트.

CONTEXT 9절: 규칙 스윕에는 항상 '아무것도 안 하는' 설정을 포함하고
기준선과 정확히 일치하는지 확인한다.
"""
import csv, hashlib, itertools, json, sys
from collections import Counter
from pathlib import Path
ROOT = Path("/workspace/DLC"); sys.path.insert(0, str(ROOT/"scripts"))
from tir_common import normalize as n

POOLS = {"A100":"outputs/tirc_hybrid3145_holdout464_vote45.jsonl",
         "V45a":"outputs/tir_v45_a100_holdout_vote45.jsonl",
         "V45r":"outputs/tir_v45_r1_holdout_vote45.jsonl",
         "V45n":"outputs/tir_v45_nc_holdout_vote45.jsonl"}
half = lambda i: int(hashlib.sha256(("split:"+i).encode()).hexdigest()[:8],16)%2==0

band = list(csv.DictReader(open(ROOT/"data/holdout/holdout464_vote45_band.csv", encoding="utf-8-sig")))
gold = {r["id"]: n(r["answer"]) for r in band}
meta = json.load(open(ROOT/"outputs/holdout464_vote45_meta.json"))
champ = {k: n(v["champ"]) for k,v in meta.items()}
risky = {k: v["nrisky"] for k,v in meta.items()}
ids = [r["id"] for r in band if gold[r["id"]] is not None and champ.get(r["id"]) is not None]

pools = {}
for k,rel in POOLS.items():
    p = ROOT/rel
    if not p.exists(): continue
    d={}
    for l in open(p):
        r=json.loads(l); c=Counter()
        for a,v in (r.get("verified_counts") or {}).items():
            a=n(a)
            if a is not None: c[a]+=v
        d[r["id"]]=c
    pools[k]=d
avail=list(pools)

def apply(cb, mc, risky_only):
    out={}
    for i in ids:
        if risky_only and risky.get(i,0)<1: out[i]=champ[i]; continue
        t=Counter()
        for k in cb: t+=pools[k].get(i,Counter())
        tp=t.most_common()
        pick=tp[0][0] if tp and tp[0][1]>=mc and (len(tp)==1 or tp[0][1]>tp[1][1]) else None
        out[i]=pick if pick is not None else champ[i]
    return out

score=lambda pr: sum(1 for i in ids if pr[i]==gold[i])
champ_pr={i:champ[i] for i in ids}
cbase=score(champ_pr)

# sanity: 발동 0 (mc 를 불가능하게 높게) -> 챔피언과 정확히 같아야 한다
sanity=apply(("V45n",), 999, True)
assert sanity==champ_pr, "SANITY FAIL: 발동 0 인데 챔피언과 다르다"
print(f"[sanity] 발동 0 설정 = 챔피언과 완전 일치 ✓  (챔피언 {cbase}/{len(ids)})")

CUR=("A100","V45n"); CUR_MC=2
cur_pr=apply(CUR,CUR_MC,True); cur=score(cur_pr)
print(f"[현행] A100+V45n mc2 risky = {cur}/{len(ids)}  (챔피언 대비 {cur-cbase:+d})")
print(f"       발동 {sum(1 for i in ids if cur_pr[i]!=champ[i])}문제\n")

rows=[]
for r_ in range(1,len(avail)+1):
    for cb in itertools.combinations(avail,r_):
        for ro in (True,False):
            for mc in range(2,3*len(cb)+1):
                pr=apply(cb,mc,ro); s=score(pr)
                if (cb,mc,ro)==(CUR,CUR_MC,True): continue
                g=sum(1 for i in ids if pr[i]!=cur_pr[i] and pr[i]==gold[i])
                rg=sum(1 for i in ids if pr[i]!=cur_pr[i] and cur_pr[i]==gold[i])
                disc=[i for i in ids if pr[i]!=cur_pr[i]]
                bw=sum(1 for i in disc if pr[i]==gold[i])
                cw=sum(1 for i in disc if cur_pr[i]==gold[i])
                cn=sum(1 for i in ids if half(i) and pr[i]==gold[i])
                cc=sum(1 for i in ids if half(i) and cur_pr[i]==gold[i])
                rows.append((s-cur,"+".join(cb),8*len(cb),mc,"risky" if ro else "전면",
                             g,rg,len(disc),cn-cc,(s-cn)-(cur-cc),bw,cw))
rows.sort(key=lambda x:(-x[0],-min(x[8],x[9])))
print("=== 현행 대비 (C-1 기준) ===")
print(f"{'조합':<18}{'N':>3}{'mc':>4}{'게이트':>7}{'델타':>6}{'gain':>6}{'reg':>5}"
      f"{'변경':>5}{'calib':>7}{'valid':>7}{'짝비교':>9}")
for r in rows[:14]:
    print(f"{r[1]:<18}{r[2]:>3}{r[3]:>4}{r[4]:>7}{r[0]:>+6}{r[5]:>6}{r[6]:>5}"
          f"{r[7]:>5}{r[8]:>+7}{r[9]:>+7}{str(r[10])+':'+str(r[11]):>9}")
print(f"\n채택 조건: 델타 >= +4 & calib/valid 양쪽 양수 & 짝비교 명확")
best=rows[0]
ok = best[0]>=4 and best[8]>0 and best[9]>0 and best[10]>best[11]
print(f"최고 후보 {best[1]} mc{best[3]} {best[4]}: 델타 {best[0]:+d}, "
      f"calib {best[8]:+d}, valid {best[9]:+d}, 짝비교 {best[10]}:{best[11]}")
print("판정:", "채택" if ok else "★ 기각 — 현행 유지")
