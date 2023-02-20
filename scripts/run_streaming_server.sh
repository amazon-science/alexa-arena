#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


killServers()
{
        echo "===== Shutting down servers. ====="
        sudo pkill node
}

trap killServers SIGINT

export ALEXA_ARENA_DIR="$HOME/AlexaArena"
cd $ALEXA_ARENA_DIR/arena/StreamingServerWebRTC
npm install
npm run build
cd ..
sudo iptables -t nat -A PREROUTING -p tcp --dport 81 -j REDIRECT --to-port 8080
sudo iptables -t nat -I OUTPUT -p tcp -d 127.0.0.1 --dport 81 -j REDIRECT --to-ports 8080

sleep 5

npm run start --prefix ./StreamingServerWebRTC -- -p 8080 -w
killServers
