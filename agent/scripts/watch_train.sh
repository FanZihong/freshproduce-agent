#!/usr/bin/env bash
# 训练守护：发现训练进程消失就自动续跑（笔记本休眠、断连、误关终端的兜底）
#
# 为什么需要它：CPU 上 v2 要跑 5 小时以上，一旦被系统挂起进程就没了；
# 等人发现再手动续跑，中间往往白等几十分钟。这个脚本每 2 分钟检查一次，
# 发现中断就自动从最新 checkpoint 接着跑。
#
# 用法（在 agent/ 目录下执行）：
#   setsid nohup bash scripts/watch_train.sh > watch_train.log 2>&1 < /dev/null &
#   tail -f watch_train.log                 # 看守护日志
#   bash scripts/watch_train.sh --stop      # 停止守护（不会杀正在跑的训练）
#
# 可用环境变量覆盖：OUT_DIR / LOG / INTERVAL / MAX_RESTART / TRAIN_ARGS
set -u

OUT_DIR="${OUT_DIR:-data/intent/models/v2}"
LOG="${LOG:-train_v2.log}"
INTERVAL="${INTERVAL:-120}"
MAX_RESTART="${MAX_RESTART:-8}"
TRAIN_ARGS="${TRAIN_ARGS:---prompt-detail catalog --max-len 0 --epochs 1 --save-steps 100 --eval-limit 0}"
PID_FILE="$OUT_DIR/.watchdog.pid"

log() { echo "[$(date '+%m-%d %H:%M:%S')] $*"; }

if [ "${1:-}" = "--stop" ]; then
  if [ -f "$PID_FILE" ]; then
    kill "$(cat "$PID_FILE")" 2>/dev/null && log "已停止守护进程"
    rm -f "$PID_FILE"
  else
    log "没有在跑的守护进程（$PID_FILE 不存在）"
  fi
  exit 0
fi

mkdir -p "$OUT_DIR"
echo $$ > "$PID_FILE"
log "守护启动：每 ${INTERVAL}s 检查一次；完成标志是 $OUT_DIR/adapter_config.json"
log "训练参数：$TRAIN_ARGS"

restarts=0
while true; do
  if [ -f "$OUT_DIR/adapter_config.json" ]; then
    log "看到 adapter_config.json，训练已完成，守护退出"
    rm -f "$PID_FILE"
    exit 0
  fi

  if pgrep -f "train_intent.py" > /dev/null 2>&1; then
    sleep "$INTERVAL"
    continue
  fi

  if [ "$restarts" -ge "$MAX_RESTART" ]; then
    log "已自动续跑 $restarts 次仍未跑完，停止守护——需要人工看一眼 $OUT_DIR 与 $LOG 末尾"
    rm -f "$PID_FILE"
    exit 1
  fi

  restarts=$((restarts + 1))
  if compgen -G "$OUT_DIR/checkpoints/checkpoint-*" > /dev/null; then
    log "训练中断（第 $restarts 次续跑）：从最新 checkpoint 继续"
    nohup python -u scripts/train_intent.py $TRAIN_ARGS --resume-from-checkpoint latest \
      --out-dir "$OUT_DIR" >> "$LOG" 2>&1 &
  else
    log "训练中断且没有 checkpoint（第 $restarts 次）：从头开始"
    nohup python -u scripts/train_intent.py $TRAIN_ARGS --out-dir "$OUT_DIR" >> "$LOG" 2>&1 &
  fi

  sleep 60
  if pgrep -f "train_intent.py" > /dev/null 2>&1; then
    log "续跑成功，训练已在运行"
  else
    log "续跑后进程没起来，等下一轮再试（先看 $LOG 末尾的报错）"
  fi
done
