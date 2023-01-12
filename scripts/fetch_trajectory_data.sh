#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


aws s3 cp s3://alexa-arena-resources/data/trajectory-data/train.json ~/AlexaArena/data/trajectory-data/train.json
aws s3 cp s3://alexa-arena-resources/data/trajectory-data/valid.json ~/AlexaArena/data/trajectory-data/valid.json
aws s3 cp s3://alexa-arena-resources/data/trajectory-data/mission_images.zip ~/AlexaArena/data/trajectory-data/mission_images.zip
unzip ~/AlexaArena/data/trajectory-data/mission_images.zip -d ~/AlexaArena/data/trajectory-data/
