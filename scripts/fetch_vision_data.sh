#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


imagespath="$HOME/AlexaArena/data/vision_data/"
mkdir -p $imagespath
aws s3 cp s3://alexa-arena-resources/data/vision-data/data1.zip $imagespath
aws s3 cp s3://alexa-arena-resources/data/vision-data/data2.zip $imagespath
aws s3 cp s3://alexa-arena-resources/data/vision-data/validation_data.zip $imagespath
aws s3 cp s3://alexa-arena-resources/data/vision-data/object_manifests.zip $HOME/AlexaArena/data/
unzip $imagespath/data1.zip -d $imagespath
unzip $imagespath/data2.zip -d $imagespath
unzip $imagespath/validation_data.zip -d $imagespath
unzip $HOME/AlexaArena/data/object_manifests.zip -d $HOME/AlexaArena/data/
