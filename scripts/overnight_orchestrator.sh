#!/usr/bin/env bash
# 2026-08-28 야간 무인 오케스트레이터 — 세션(SSH/VSCode) 생존과 무관하게 완주한다.
# 체인: 서버 대기 → 스크리닝 기동 → (판정 체인 종료 대기) → 스크리닝 완료 대기
#       → 밴드 선별 → 서버 PID 종료 → GRPO 본 학습 기동
# 실행: setsid nohup bash scripts/overnight_orchestrator.sh > logs/overnight_orchestrator.log 2>&1 < /dev/null &
set -u
R=/workspace/DLC
V=/workspace/venv-vllm/bin/python
cd $R
say(){ echo "[$(TZ=Asia/Seoul date +%H:%M:%S)] $*"; }
up(){ curl -s -m 3 http://localhost:8000/v1/models 2>/dev/null | grep -q hybrid3145; }

SCREEN_OUT=outputs/grpo_screen4000_hybrid3145_n8.jsonl
POOL_CSV=data/processed/grpo_screen_pool4000.csv
BAND=data/processed/grpo_passrate_scaleup.jsonl
JUDGE_PID=${JUDGE_PID:-3448}
VLLM_PID=${VLLM_PID:-3443}

say "=== 1) 서버 대기 (최대 30분) ==="
for _ in $(seq 1 120); do up && break; sleep 15; done
up || { say "서버 기동 실패 — 중단"; exit 1; }
say "서버 up"

say "=== 2) 스크리닝 기동 (idempotent) ==="
if pgrep -f screen_grpo_passrate.py >/dev/null; then
  say "이미 실행 중 — 건너뜀"
else
  setsid nohup $V scripts/screen_grpo_passrate.py \
    --input $POOL_CSV --output $SCREEN_OUT \
    --model hybrid3145 --num-samples 8 --temperature 0.7 --top-p 0.95 \
    --max-new-tokens 1024 --request-workers 48 --seed 20260916 --resume \
    > logs/grpo_screen4000.log 2>&1 < /dev/null &
  say "스크리닝 pid=$!"
fi

say "=== 3) 스크리닝 완료 대기 (죽으면 --resume 재기동, 최대 3회) ==="
RESTARTS=0
while true; do
  sleep 60
  DONE=$([ -f $SCREEN_OUT ] && wc -l < $SCREEN_OUT || echo 0)
  if [ "$DONE" -ge 4000 ]; then say "스크리닝 완료: $DONE/4000"; break; fi
  if ! pgrep -f screen_grpo_passrate.py >/dev/null; then
    if ! up; then say "서버도 죽음 — 재기동 없이 중단(수동 개입 필요), 진행분 $DONE"; break; fi
    RESTARTS=$((RESTARTS+1))
    [ $RESTARTS -gt 3 ] && { say "재기동 3회 초과 — 진행분 $DONE 으로 진행"; break; }
    say "스크리닝 프로세스 사망(진행 $DONE) — resume 재기동 #$RESTARTS"
    setsid nohup $V scripts/screen_grpo_passrate.py \
      --input $POOL_CSV --output $SCREEN_OUT \
      --model hybrid3145 --num-samples 8 --temperature 0.7 --top-p 0.95 \
      --max-new-tokens 1024 --request-workers 48 --seed 20260916 --resume \
      > logs/grpo_screen4000_r$RESTARTS.log 2>&1 < /dev/null &
  fi
done

say "=== 4) 판정 체인 종료 대기 (서버를 아직 쓰고 있을 수 있음) ==="
while kill -0 $JUDGE_PID 2>/dev/null; do sleep 30; done
say "판정 체인 종료 — 결과 tail:"
tail -12 logs/teacher_judge.log

say "=== 5) 밴드 선별 (n_correct 2~6/8) ==="
/usr/bin/python3 scripts/build_grpo_passrate_pool.py \
  --screen $SCREEN_OUT --pool-csv $POOL_CSV --output $BAND \
  || { say "밴드 선별 실패 — 중단"; exit 1; }
NBAND=$(wc -l < $BAND)
say "밴드 풀: $NBAND 문제"
[ "$NBAND" -lt 300 ] && { say "밴드 300 미만 — GRPO 기동 보류(사용자 판단 대기)"; exit 0; }

say "=== 6) vLLM 서버 종료 (PID $VLLM_PID, cmdline 검증 후) ==="
if kill -0 $VLLM_PID 2>/dev/null && grep -q vllm /proc/$VLLM_PID/cmdline 2>/dev/null; then
  kill $VLLM_PID
  for _ in $(seq 1 24); do kill -0 $VLLM_PID 2>/dev/null || break; sleep 5; done
  kill -0 $VLLM_PID 2>/dev/null && kill -9 $VLLM_PID
  say "서버 종료 완료"
else
  say "서버 PID $VLLM_PID 없음/불일치 — 종료 생략"
fi
sleep 10

say "=== 7) GRPO 본 학습 기동 (17절 스케일업: steps 800, LR 2e-6, g8) ==="
setsid nohup env PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True \
  /usr/bin/python3 scripts/train_grpo_qlora.py \
  --data $BAND \
  --adapter-path checkpoints/hybrid_3145_r8_qv_lr2e6_e1/final_adapter \
  --output-dir checkpoints/grpo_3145_scaleup_r8_qv_lr2e6_steps800_g8 \
  --max-steps 800 --max-minutes 720 --learning-rate 2e-6 \
  --batch-size 8 --gradient-accumulation 2 --num-generations 8 \
  --max-completion-length 512 --temperature 0.8 --top-p 0.95 --beta 0.005 \
  --save-steps 50 --save-total-limit 20 --seed 20260917 \
  > logs/grpo_scaleup_train.log 2>&1 < /dev/null &
say "GRPO pid=$! — step당 ~45초, 800 step ≈ 10h, 50 step 마다 체크포인트"
say "=== 오케스트레이터 완료 (GRPO 는 백그라운드 계속) ==="
