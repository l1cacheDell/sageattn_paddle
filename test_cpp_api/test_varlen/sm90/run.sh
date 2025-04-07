#!/bin/bash

export CUDA_VISIBLE_DEVICES=7

compute-sanitizer \
python test_sm90_varlen_bsz1.py &
PID=$!

echo "启动的进程PID: $PID"
echo "如果要终止该进程，可执行: kill $PID"

# 定义 cleanup 函数，在脚本退出时 kill 子进程
cleanup() {
    echo "捕获到 Ctrl+C，正在终止进程 $PID..."
    echo "如果要终止该进程，可执行: kill -9 $PID"
    kill $PID 2>/dev/null  # 2>/dev/null 是为了避免 kill 不存在的进程时报错
    exit 1
}



# 设置 trap，捕获 SIGINT (Ctrl+C) 和 EXIT 信号
trap cleanup SIGINT EXIT

# 等待进程结束
wait $PID