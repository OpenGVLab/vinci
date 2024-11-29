#!/bin/bash

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

HOSTNAME_ARG=""
PULL_ARG=""
ACTION="start"

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --hostname)
            HOSTNAME_ARG="$2"; shift ;;
        --pull)
            PULL_ARG="true" ;;
        start|stop|restart|clean|ps)
            ACTION="$1" ;;
        *)
            echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

if [ -n "$HOSTNAME_ARG" ]; then
    LAN_IP="$HOSTNAME_ARG"
else
    LAN_IP=$(hostname -I | awk '{print $1}')
fi

echo "LAN IP: $LAN_IP"

export CANDIDATE="$LAN_IP"

# 起停 docker-compose 服务
cd "$SCRIPT_DIR"

# 是否拉取最新镜像
if [ "$PULL_ARG" == "true" ]; then
    echo "Pulling latest images..."
    docker compose pull
fi

case "$ACTION" in
    start)
        echo "Starting docker-compose services..."
        docker compose up -d
        ;;
    stop)
        echo "Stopping docker-compose services..."
        docker compose down
        ;;
    restart)
        echo "Restarting docker-compose services..."
        docker compose down
        docker compose up -d
        ;;
    ps)
        echo "Listing docker-compose services status..."
        docker compose ps
        ;;
    clean)
        echo "Stopping docker-compose services and cleaning up..."
        docker compose down
        if [ -d "$SCRIPT_DIR/.cache" ]; then
            rm -rf "$SCRIPT_DIR/.cache"
            echo "Removed .cache directory."
        else
            echo "No .cache directory found."
        fi
        ;;
    *)
        echo "No valid action specified. Use 'start' or 'stop'."
        ;;
esac
