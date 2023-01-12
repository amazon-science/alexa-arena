# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


from flask import abort
import json
import logging
import threading
import time
import socket
import sys

logging.getLogger("werkzeug").setLevel(logging.ERROR)


class ArenaController:
    def __init__(self, host='127.0.0.1'):
        self.last_rate_timestamp = time.time()
        self.frame_counter = 0
        self.debug_frames_per_interval = 50
        self.UnityWSPath = host
        self.UnityWSPort = 5000
        self.isSocketOpen = False
        self.ws = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.resultJSON = list()
        self.currentBatchNum = 0
        self.currentRespNum = 0
        self.isUnityConnected = threading.Event()

    def interact(self, actions):
        batchNum = self.currentBatchNum
        self.currentBatchNum += 1
        self.wsSend(actions)
        JSONResponse = dict()
        ticks = 0
        while not JSONResponse:
            if not self.resultJSON:
                time.sleep(0.1)
                ticks += 1
                if ticks >= 6000:
                    abort(408)
                    resp.set_status(408)
                    return resp
                if not self.isSocketOpen:
                    abort(404)
                    resp.set_status(404)
                    return resp
                continue
            else:
                for JSON in self.resultJSON:
                    if JSON["batchNum"] == batchNum:
                        JSONResponse = JSON
                        self.resultJSON.remove(JSON)

        resp = json.dumps(JSONResponse)
        return resp

    def handle_init(self, init_request):
        print("Received initialize message. Sending it to Unity application to bring up for play.")
        self.wsSend(init_request)
        return

    def start(self):
        self.isUnityConnected.set()

        self.ws_listen_thread = threading.Thread(target=self.wsListen)
        self.ws_listen_thread.daemon = True
        self.ws_listen_thread.start()

        self.ws_monitor_thread = threading.Thread(target=self.wsMonitor)
        self.ws_monitor_thread.daemon = True
        self.ws_monitor_thread.start()

        print('Listener and monitor threads successfully started.')

    def wsConnect(self):
        print('Awaiting connection to Unity instance')
        self.isSocketOpen = False
        self.currentBatchNum = 0
        self.currentRespNum = 0
        self.resultJSON.clear()
        counter = 0
        print("self.UnityWSPath: ", self.UnityWSPath)
        # Loop until a connection is made
        while self.isUnityConnected.is_set():
            time.sleep(0.1)

            self.ws.close()
            self.ws = socket.socket(socket.AF_INET, socket.SOCK_STREAM)

            result = self.ws.connect_ex((self.UnityWSPath, self.UnityWSPort))

            if result == 0:
                self.isSocketOpen = True
                print('Connection established')
                return
            else:
                print("Could not connect to RG unity instance: ", result)
                if counter == 250:
                    print("Tried to connect to unity for 250 times. Stopping the controller.")
                    self.isUnityConnected.clear()
            counter += 1
        return

    # Runs on its own thread listening for data from Unity
    def wsListen(self):
        while self.isUnityConnected.is_set():
            if self.isSocketOpen:
                try:
                    sizeInBytes = self.ws.recv(4)
                    if not sizeInBytes:
                        self.isSocketOpen = False
                        print('Connection lost during listener thread loop')
                        continue

                    size = int.from_bytes(bytes=sizeInBytes, byteorder=sys.byteorder, signed=False)
                    bytesReceived = 0
                    dataBuffer = bytearray()

                    while bytesReceived < size:
                        dataBuffer += self.ws.recv(size - bytesReceived)
                        bytesReceived = len(dataBuffer)

                    jsonData = str(dataBuffer, encoding = 'UTF-8')
                    #print(jsonData + '\n')

                    JSONPacket = json.loads(jsonData)
                    JSONPacket["batchNum"] = self.currentRespNum
                    self.currentRespNum += 1
                    self.resultJSON.append(JSONPacket)

                except socket.error as e:
                    print('Exception during read')
                    if e.errno == socket.errno.ECONNRESET:
                        self.isSocketOpen = False
                    else:
                        raise
            else:
                time.sleep(0.1)
        print("Listen thread ends")

    def wsSend(self, jsonCommand):
        isDataSent = False
        while not isDataSent:
            if self.isSocketOpen:
                encodedString = json.dumps(jsonCommand).encode(encoding='UTF-8')
                encodedBytesToSend = len(encodedString).to_bytes(4, sys.byteorder)

                try:
                    bytesSent = 0
                    bytesSent += self.ws.send(encodedBytesToSend)
                    bytesSent += self.ws.send(encodedString)
                    if bytesSent > 0:
                        print(str(bytesSent) + ' of expected ' + str(
                            len(encodedString) + 4) + ' bytes sent.\n' + json.dumps(jsonCommand))
                        isDataSent = True

                except socket.error as e:
                    if e.errno == socket.errno.ECONNRESET:
                        self.isSocketOpen = False
                    else:
                        raise

            time.sleep(0.1)

    def wsMonitor(self):
        while self.isUnityConnected.is_set():
            try:
                self.ws.send(bytearray(0))
            except:
                self.isSocketOpen = False
            if not self.isSocketOpen:
                self.wsConnect()

            time.sleep(1.0)
        self.isSocketOpen = False
        print("Monitor thread ends")

    def stop(self):
        self.isUnityConnected.clear()
        print("Unity exe disconnected successfully")

    def get_connection_status(self):
        return self.isUnityConnected.is_set()
