#!/bin/bash

wget -P /tmp/ "https://alexa-arena-executable.s3.us-west-1.amazonaws.com/Arena.zip"
unzip /tmp/Arena.zip -d ~/AlexaArena/arena
rm -f /tmp/Arena.zip