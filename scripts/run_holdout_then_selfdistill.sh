#!/usr/bin/env bash
# 순서 바로잡기: **홀드아웃 판정 먼저, 831 생성은 나중**  (CONTEXT C-4)
#
# 오늘 이 순서를 세 번 어겼다. 831 에는 정답이 없어서 규칙을 판정할 수 없는데
# 미검증 32샘플 규칙을 위해 831(16분)부터 돌렸다. 홀드아웃87 은 6분이고 정답이 있다.
#
#   1) 홀드아웃87 에 새 풀 2개 생성 (설정 다양성 확보: 리페어2 / adaptive)
#   2) 8~40샘플 조합 스윕 -> 24샘플 mc2(+9)를 넘는 게 있는지 판정
#   3) 그 다음에야 자기증류 학습
#
# ⚠️ 프로세스 종료는 PID 로만 (pkill -f 는 자기 자신을 죽인다, CONTEXT 14절)
set -u
R=/workspace/DLC
V=/workspace/venv-vllm/bin/python
H=$R/data/holdout/holdout464_vote3.csv
say(){ echo "[$(TZ=Asia/Seoul date +%H:%M)] $*"; }
up(){ curl -s -m 3 -o /dev/null http://localhost:8000/v1/models; }

say "=== 1) 홀드아웃87 새 풀 생성 ==="
up || { say "서버 DOWN — 중단"; exit 1; }

# R2: 리페어 2라운드 (기존 R1 과 설정이 다르다 = 설정 다양성)
o=$R/outputs/tir_r2_holdout464_vote3.jsonl
if [ ! -f "$o" ]; then
  say "R2 (리페어 2라운드) 생성"
  $V $R/scripts/tir_repair_client.py --input "$H" --output "$o" \
    --model hybrid3145 --num-samples 8 --repair-rounds 2 --nocode-retries 1 \
    --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed 20260908 \
    > $R/logs/holdout_r2.log 2>&1 && tail -2 $R/logs/holdout_r2.log || say "R2 실패"
fi

# NC3: NC 설정 새 시드 (시드 다양성 — 대조군)
o=$R/outputs/tir_nc3_holdout464_vote3.jsonl
if [ ! -f "$o" ]; then
  say "NC3 (NC 설정 새 시드) 생성"
  $V $R/scripts/tir_repair_client.py --input "$H" --output "$o" \
    --model hybrid3145 --num-samples 8 --repair-rounds 1 --nocode-retries 1 \
    --exec-timeout 60 --exec-workers 32 --request-workers 32 --seed 20260909 \
    > $R/logs/holdout_nc3.log 2>&1 && tail -2 $R/logs/holdout_nc3.log || say "NC3 실패"
fi

say "=== 2) 조합 스윕 판정 ==="
python3 $R/scripts/sweep_pools_holdout.py 2>&1 | tee $R/logs/pool_sweep.log

say "=== 3) 자기증류 이어서 ==="
setsid nohup bash $R/scripts/run_selfdistill_after_nc.sh \
  > $R/logs/selfdistill_chain4.log 2>&1 < /dev/null &
say "자기증류 연쇄 재개 (PID $!)"
