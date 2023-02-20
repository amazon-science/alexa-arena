# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1
import os
HOME_PATH = os.getenv('HOME')

# Paths
train_data = os.path.join(HOME_PATH, 'AlexaArena/data/trajectory-data/train.json')
validation_data = os.path.join(HOME_PATH, 'AlexaArena/data/trajectory-data/valid.json')
train_numpy = os.path.join(HOME_PATH, 'AlexaArena/data/trajectory-data/nlg_commands_train.npy')
validation_numpy = os.path.join(HOME_PATH, 'AlexaArena/data/trajectory-data/nlg_commands_val.npy')

# Precomputed channel wise image means and stds for model
means = [0.48145466, 0.4578275, 0.40821073]
stds = [0.26862954, 0.26130258, 0.27577711]
