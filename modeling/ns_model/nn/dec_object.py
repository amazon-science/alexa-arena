# Modified from https://github.com/alexpashevich/E.T./blob/master/alfred/nn/dec_object.py - Licensed under the
# MIT License.
#
# Modifications Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.


import os
import json
import torch
from torch import nn
from torch.nn import functional as F


class ObjectClassifier(nn.Module):
    '''
    object classifier module (a single FF layer)
    '''
    def __init__(self, input_size):
        super().__init__()
        with open(os.getenv('ALEXA_ARENA_DIR') + "/data/vision-data/class_to_idx.json", "r") as f:
            vocab_obj = json.load(f)
        num_classes = len(vocab_obj)
        self.linear = nn.Linear(input_size, num_classes)

    def forward(self, x):
        out = self.linear(x)
        return out
