#!/bin/bash
# VLM 服务定时测试脚本
# 每隔 5 秒调用一次 test_vlm.py

URL="${URL:-http://192.168.4.52:8080}"
PROMPT="${PROMPT:-你好，请介绍一下自己}"
INTERVAL="${INTERVAL:-5}"
TIMES=0

echo "============================================================"
echo "🤖 VLM 服务定时测试"
echo "============================================================"
echo "📍 服务器: $URL"
echo "💭 Prompt: $PROMPT"
echo "⏱️  间隔: ${INTERVAL}秒"
echo "💡 按 Ctrl+C 停止"
echo "============================================================"
echo ""

trap 'echo ""; echo "🛑 测试停止"; exit 0' INT

while true; do
    ((TIMES++))
    echo "--- 第 $TIMES 次测试 ($(date '+%Y-%m-%d %H:%M:%S')) ---"

    python test_vlm.py --prompt "$PROMPT" --url "$URL"

    if [ $? -ne 0 ]; then
        echo "⚠️ 测试失败，等待 ${INTERVAL}秒 后重试..."
    fi

    echo ""
    sleep "$INTERVAL"
done
