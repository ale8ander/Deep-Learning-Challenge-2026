#!/usr/bin/env bash
# ck150 base 교체 스택 전이 실측 — 표수<=3 밴드 87문제의 TIR 풀을 ck150 으로 재생성해
# 현행 배포 규칙(16샘플 mc3, 챔피언 fallback)과 같은 잣대로 비교한다. 학습과 공존(서버 0.45).
set -u
R=/workspace/DLC
V=/workspace/venv-vllm/bin/python
CK=$R/checkpoints/grpo_3145_scaleup_r8_qv_lr2e6_steps800_g8/checkpoint-150
say(){ echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] $*"; }
up(){ curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q hybrid3145; }
cd $R

say "=== 1) 서버 대기 ==="
for _ in $(seq 1 100); do up && break; sleep 15; done
up || { say "서버 기동 실패"; exit 1; }
curl -s -X POST http://localhost:8000/v1/load_lora_adapter -H 'Content-Type: application/json' \
  -d "{\"lora_name\":\"ck150\",\"lora_path\":\"$CK\"}"; echo

say "=== 2) ck150 NC형 TIR 8샘플 — seed 20260918 (게이트 B 와 동일 설정) ==="
$V scripts/tir_repair_client.py \
  --input data/holdout/holdout464_vote3.csv \
  --output outputs/ck150_tir8_nc_holdout87_s18.jsonl \
  --model ck150 --num-samples 8 --repair-rounds 1 --nocode-retries 1 \
  --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed 20260918 2>&1 | tail -2

say "=== 3) ck150 NC형 TIR 8샘플 — seed 20260919 (두 번째 풀) ==="
$V scripts/tir_repair_client.py \
  --input data/holdout/holdout464_vote3.csv \
  --output outputs/ck150_tir8_nc_holdout87_s19.jsonl \
  --model ck150 --num-samples 8 --repair-rounds 1 --nocode-retries 1 \
  --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed 20260919 2>&1 | tail -2

say "=== 4) 판정 — 현행 배포 규칙과 동일 잣대 (unique-mode>=mc, 챔피언 fallback) ==="
/usr/bin/python3 - <<'PY'
import csv, hashlib, json, sys
from collections import Counter
sys.path.insert(0,"/workspace/DLC/scripts")
from tir_common import normalize as n
R="/workspace/DLC/"
POOLS={"A100":"outputs/tir_sc8_holdout464_vote3_to60.jsonl",
       "NC1":"outputs/tir_repair_nocode_holdout464_vote3.jsonl",
       "CK18":"outputs/ck150_tir8_nc_holdout87_s18.jsonl",
       "CK19":"outputs/ck150_tir8_nc_holdout87_s19.jsonl"}
def half(i): return int(hashlib.sha256(("split:"+i).encode()).hexdigest()[:8],16)%2==0
def load(rel):
    d={}
    for l in open(R+rel):
        r=json.loads(l); c=Counter()
        for a,v in (r.get("verified_counts") or {}).items():
            a=n(a)
            if a is not None: c[a]+=v
        d[r["id"]]=c
    return d
gold={r["id"]: n(r["answer"]) for r in csv.DictReader(open(R+"data/holdout/holdout464_vote3.csv",encoding="utf-8-sig"))}
champ={}
for l in open(R+"outputs/champion_holdout464_equivalent.jsonl"):
    r=json.loads(l); champ[r["id"]]=n(r.get("prediction"))
ids=[i for i in gold if gold[i] is not None and i in champ]
pools={k:load(v) for k,v in POOLS.items()}
base=sum(1 for i in ids if champ[i]==gold[i])
CH464=373
print(f"밴드 {len(ids)}문제 / 챔피언(656등가) 밴드 {base} / holdout464 전체 {CH464}\n")
def apply(cb,mc):
    out={}
    for i in ids:
        t=Counter()
        for k in cb: t+=pools[k].get(i,Counter())
        tp=t.most_common()
        pick=tp[0][0] if tp and tp[0][1]>=mc and (len(tp)==1 or tp[0][1]>tp[1][1]) else None
        out[i]=pick if pick is not None else champ[i]
    return out
def rep(label,cb,mc):
    pr=apply(cb,mc)
    s=sum(1 for i in ids if pr[i]==gold[i])
    g=sum(1 for i in ids if pr[i]!=champ[i] and pr[i]==gold[i])
    rg=sum(1 for i in ids if pr[i]!=champ[i] and champ[i]==gold[i])
    cn=sum(1 for i in ids if half(i) and pr[i]==gold[i])
    cb_=sum(1 for i in ids if half(i) and champ[i]==gold[i])
    orc=sum(1 for i in ids if any(gold[i] in pools[k].get(i,{}) for k in cb))
    print(f"{label:<28} 델타 {s-base:+d} → 464환산 {CH464+s-base}  gain {g} reg {rg}  calib {cn-cb_:+d} valid {(s-cn)-(base-cb_):+d}  오라클 {orc}")
    return s-base
print("— 단독 8샘플 mc2 —")
rep("NC1 (hybrid, 현행 재료)",("NC1",),2)
rep("CK18 (ck150)",("CK18",),2)
rep("CK19 (ck150)",("CK19",),2)
print("\n— 16샘플 mc3 (현행 배포 규칙) —")
d_cur=rep("A100+NC1 = 현행656",("A100","NC1"),3)
d_swap=rep("CK18+CK19 = ck150 교체",("CK18","CK19"),3)
print("\n— 참고: 혼합(교체 아님, 채택 불가·정보용) —")
rep("A100+NC1+CK18+CK19 mc4",("A100","NC1","CK18","CK19"),4)
print(f"\n결론: ck150 교체본 464환산 {CH464+d_swap} vs 현행 {CH464+d_cur} (진짜 배포 기준 373)")
print("주의: C-2 — 이 밴드 홀드아웃 델타는 Public 과 부호가 다를 수 있음. 채택 전 Public 검증 필요.")
PY
say "=== 완료 ==="
