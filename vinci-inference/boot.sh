#!/bin/bash

# 获取脚本所在目录
SCRIPT_DIR=$(dirname "$(realpath "$0")")

# 加载环境变量文件
if [ -f "$SCRIPT_DIR/.env" ]; then
    source "$SCRIPT_DIR/.env"
else
    echo ".env file not found in $SCRIPT_DIR. Exiting."
    exit 1
fi


# Conda 环境名称
CONDA_ENV=vinci

# 默认的 CUDA 设备
DEFAULT_CUDA_VISIBLE_DEVICES="0"
DEFAULT_RUNNING_LANGUAGE='chn'
DEFAULT_VERSION="v1"

# Python 启动命令
COMMAND="python vinci-inference/app/main.py"
LOG_FILE="vinci_inference.log"
PID_FILE="/tmp/.vinci/vinci_inference.pid"

# 确保目录存在
mkdir -p "$(dirname "$PID_FILE")"

# 函数：启动服务
start_service() {
    export CUDA_VISIBLE_DEVICES=$1
    export RUNNING_LANGUAGE=$2
    export VERSION=$3

    echo "Activating conda environment: $CONDA_ENV"
    source $(conda info --base)/etc/profile.d/conda.sh
    conda activate $CONDA_ENV

    echo "Starting service with CUDA_VISIBLE_DEVICES=$CUDA_VISIBLE_DEVICES..."
    $COMMAND &
    echo $! > $PID_FILE
    echo "Service started with PID $(cat $PID_FILE)"
}

# 函数：停止服务
stop_service() {
    if [ -f $PID_FILE ]; then
        PID=$(cat $PID_FILE)
        echo "Stopping service with PID $PID..."
        kill $PID
        rm -f $PID_FILE
        echo "Service stopped."
    else
        echo "Service is not running."
    fi
}

# 函数：重启服务
restart_service() {
    echo "Restarting service..."
    stop_service
    start_service $1 $2
    echo "Service restarted."
}

# 主逻辑
CUDA_VISIBLE_DEVICES=$DEFAULT_CUDA_VISIBLE_DEVICES
RUNNING_LANGUAGE=$DEFAULT_RUNNING_LANGUAGE
VERSION=$DEFAULT_VERSION

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda) CUDA_VISIBLE_DEVICES="$2"; shift ;;
        --version) VERSION="$2"; shift ;;
        --language) RUNNING_LANGUAGE="$2"; shift ;;
        start) COMMAND_ACTION="start" ;;
        stop) COMMAND_ACTION="stop" ;;
        restart) COMMAND_ACTION="restart" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

case "$COMMAND_ACTION" in
    start)
        start_service $CUDA_VISIBLE_DEVICES $RUNNING_LANGUAGE $VERSION
        ;;
    stop)
        stop_service
        ;;
    restart)
        restart_service $CUDA_VISIBLE_DEVICES $RUNNING_LANGUAGE $VERSION
        ;;
    *)
        echo "Usage: $0 {start|stop|restart} [--cuda <CUDA_VISIBLE_DEVICES>] [--language chn/eng] [--version v0/v1]"
        exit 1
        ;;
esac

exit 0
