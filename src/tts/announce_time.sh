#!/bin/bash
# 每分钟报时脚本 + 每5分钟播报累计时间

TTS_SCRIPT="/mnt/apfs/workplace/learnPartner/src/tts/test_tts.py"

if [ ! -f "$TTS_SCRIPT" ]; then
    echo "❌ TTS脚本不存在: $TTS_SCRIPT"
    exit 1
fi

# 报时函数
announce() {
    TIME=$(date "+%H点%M分%S秒")
    echo "[$(date '+%H:%M:%S')] 🕰️  现在时间是$TIME"
    python3 "$TTS_SCRIPT" --text "现在时间是$TIME" --quiet 2>/dev/null
}

# 播报累计时间
announce_elapsed() {
    ELAPSED_MINUTES=$1
    echo "[$(date '+%H:%M:%S')] ⏱️  距离开始计时已经${ELAPSED_MINUTES}分钟"
    python3 "$TTS_SCRIPT" --text "距离开始计时已经${ELAPSED_MINUTES}分钟" --quiet 2>/dev/null
}

echo "🕐 报时服务启动 (按Ctrl+C停止)"
echo ""

# 记录开始时间
START_TIME=$(date +%s)
START_MINUTE=$(date +%M)

# 立即报时一次
announce

# 循环：倒计时60秒 -> 报时
while true; do
    for i in {60..1}; do
        ELAPSED_MINUTES=$(( ($(date +%s) - START_TIME) / 60 ))
        printf "\r⏱️  倒计时: %02d秒 [已运行: %d分钟]" "$i" "$ELAPSED_MINUTES"
        sleep 1
    done
    echo ""
    announce

    # 检查是否过了5分钟
    CURRENT_MINUTE=$(date +%M)
    ELAPSED_MINUTES=$(( ($(date +%s) - START_TIME) / 60 ))

    # 每5分钟播报一次累计时间（5, 10, 15, 20...）
    if [ $((ELAPSED_MINUTES % 5)) -eq 0 ] && [ "$ELAPSED_MINUTES" -gt 0 ]; then
        announce_elapsed $ELAPSED_MINUTES
    fi
done