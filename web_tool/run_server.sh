#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


killServers()
{
    echo "===== Shutting down servers. ====="
    sudo pkill python
    sudo pkill Arena
    if [ -f "/tmp/.X1-lock" ]; then
        echo "Shutting down X server for Display:1"
        sudo kill -9 $(cat /tmp/.X1-lock)
    fi
}
trap killServers SIGINT

export PLATFORM="Linux"
export ALEXA_ARENA_DIR="$HOME/AlexaArena"
export PYTHONPATH="${PYTHONPATH}:${ALEXA_ARENA_DIR}"
export CDF_DIR_PATH="${ALEXA_ARENA_DIR}/data/CDFs"
export ARENA_PATH="$ALEXA_ARENA_DIR/arena/Linux/Arena.x86_64"
sudo chmod -R 755 $ALEXA_ARENA_DIR/arena/Linux
chmod 777 $ARENA_PATH
mkdir -p $ALEXA_ARENA_DIR/logs
export UNITY_LOG_PATH="$ALEXA_ARENA_DIR/logs/unity_logs.log"
echo "====== Starting X Server ======"
sudo /usr/bin/X :1 &
sleep 5
echo "====== Starting backend server ======"
python3 server.py
killServers
