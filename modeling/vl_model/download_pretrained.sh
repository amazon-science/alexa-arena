#!/bin/bash
# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


mkdir pretrained
# Original model download for reproducibility
# Obtained from https://openaipublic.azureedge.net/clip/models/afeb0e10f9e5a86da6080e35cf09123aca3b358a0c3e3b6c78a7b63bc04b6762/RN50.pt
aws s3 cp s3://alexa-arena-resources/model-artifacts/vl-model/public-pretrained-resources/RN50.pt pretrained/RN50.pt --no-sign-request

# Obtained from https://github.com/openai/CLIP/
# original LICENSE: MIT License https://github.com/openai/CLIP/blob/main/LICENSE
aws s3 cp s3://alexa-arena-resources/model-artifacts/vl-model/public-pretrained-resources/bpe_simple_vocab_16e6.txt.gz data_generators/bpe_simple_vocab_16e6.txt.gz --no-sign-request
