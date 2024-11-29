#!/bin/bash

CUDA="0"
RUNNING_LANGUAGE='chn'

while [[ "$#" -gt 0 ]]; do
    case $1 in
        --cuda) CUDA="$2"; shift ;;
        --language) RUNNING_LANGUAGE="$2"; shift ;;
        start) COMMAND_ACTION="start" ;;
        stop) COMMAND_ACTION="stop" ;;
        restart) COMMAND_ACTION="restart" ;;
        *) echo "Unknown parameter passed: $1"; exit 1 ;;
    esac
    shift
done

cd vinci-local/docker
./boot.sh "$COMMAND_ACTION"
cd ../..
./vinci-inference/boot.sh --cuda "$CUDA" --language "$RUNNING_LANGUAGE" "$COMMAND_ACTION"
