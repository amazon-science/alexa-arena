# Modified from https://github.com/alexpashevich/E.T./blob/master/alfred/utils/helper_util.py - Licensed under the
# MIT License.
#
# Modifications Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.


import unicodedata
import re
import torch


def collate_fn(batch_list):
    batch_dict = {}

    batch_dict['frames'] = [b['frames'] for b in batch_list]
    batch_dict['lang'] = [b['lang'] for b in batch_list]
    batch_dict['action'] = [b['action'] for b in batch_list]
    batch_dict['object'] = [b['object'] for b in batch_list]
    batch_dict['action_valid_interact'] = [b['action_valid_interact'] for b in batch_list]
    batch_dict['mission_id'] = [b['mission_id'] for b in batch_list]
    return batch_dict


def unicodeToAscii(s):
    return ''.join(
        c for c in unicodedata.normalize('NFD', s)
        if unicodedata.category(c) != 'Mn'
    )

def normalizeString(s):
    s = unicodeToAscii(s.lower().strip())
    s = re.sub(r"([.!?])", r" \1", s)
    s = re.sub(r"[^a-zA-Z.!?<>]+", r" ", s)
    return s


class DataParallel(torch.nn.DataParallel):
    """
    Allow nn.DataParallel to call model's attributes.
    """
    def __getattr__(self, name):
        try:
            return super().__getattr__(name)
        except AttributeError:
            return getattr(self.module, name)