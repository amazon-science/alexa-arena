#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


REPO_ROOT=$HOME
IMAGES_PATH=$REPO_ROOT/AlexaArena/data/image_data/
mkdir -p $imagespath
aws s3 cp s3://alexa-arena-resources/DATA_LICENSE $IMAGES_PATH --no-sign-request
aws s3 cp s3://alexa-arena-resources/data/vision-data/data1.zip $IMAGES_PATH --no-sign-request
aws s3 cp s3://alexa-arena-resources/data/vision-data/data2.zip $IMAGES_PATH --no-sign-request
aws s3 cp s3://alexa-arena-resources/data/vision-data/validation_data.zip $IMAGES_PATH --no-sign-request
aws s3 cp s3://alexa-arena-resources/data/vision-data/object_manifests.zip $HOME/AlexaArena/data/ --no-sign-request
unzip $IMAGES_PATH/data1.zip -d $IMAGES_PATH
unzip $IMAGES_PATH/data2.zip -d $IMAGES_PATH
unzip $IMAGES_PATH/validation_data.zip -d $IMAGES_PATH
unzip $REPO_ROOT/AlexaArena/data/object_manifests.zip -d $REPO_ROOT/AlexaArena/data/
