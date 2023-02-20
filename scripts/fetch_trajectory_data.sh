#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1

aws s3 cp s3://alexa-arena-resources/DATA_LICENSE ~/AlexaArena/data/DATA_LICENSE --no-sign-request
aws s3 cp s3://alexa-arena-resources/data/trajectory-data/train.json ~/AlexaArena/data/trajectory-data/train.json --no-sign-request
aws s3 cp s3://alexa-arena-resources/data/trajectory-data/valid.json ~/AlexaArena/data/trajectory-data/valid.json --no-sign-request
aws s3 cp s3://alexa-arena-resources/data/trajectory-data/mission_images.zip ~/AlexaArena/data/trajectory-data/mission_images.zip --no-sign-request
unzip ~/AlexaArena/data/trajectory-data/mission_images.zip -d ~/AlexaArena/data/trajectory-data/
