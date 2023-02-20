#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


aws s3 cp s3://alexa-arena-resources/model-artifacts/ns-model/1/ ~/AlexaArena/logs/ns_model_checkpt/1/ --recursive --no-sign-request
