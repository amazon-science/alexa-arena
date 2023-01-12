import os
import random
import time
import numpy as np


def makedirs(directory):
    os.makedirs(directory, exist_ok=True)


def atomic_write(path, data):
    tmp_path = "-".join([path, str(time.time()), str(random.random())])
    mode = "w"

    if type(data) is bytes:
        mode = "wb"

    with open(tmp_path, mode) as f:
        f.write(data)
    os.rename(tmp_path, path)


DETECTION_SCREEN_WIDTH = 300
DETECTION_SCREEN_HEIGHT = 300


def decompress_mask(compressed_mask):
    '''
    decompress compressed mask array  alfred todo: refactoring
    '''
    mask = np.zeros((DETECTION_SCREEN_WIDTH, DETECTION_SCREEN_HEIGHT))
    for start_idx, run_len in compressed_mask:
        for idx in range(start_idx, start_idx + run_len):
            mask[idx // DETECTION_SCREEN_WIDTH, idx % DETECTION_SCREEN_HEIGHT] = 1
    return mask


def compress_mask(seg_mask):
    '''
    compress mask array alfred todo: refactoring
    '''
    run_len_compressed = []  # list of lists of run lengths for 1s, which are assumed to be less frequent.
    idx = 0
    curr_run = False
    run_len = 0
    for x_idx in range(len(seg_mask)):
        for y_idx in range(len(seg_mask[x_idx])):
            if seg_mask[x_idx][y_idx] == 1 and not curr_run:
                curr_run = True
                run_len_compressed.append([idx, None])
            if seg_mask[x_idx][y_idx] == 0 and curr_run:
                curr_run = False
                run_len_compressed[-1][1] = run_len
                run_len = 0
            if curr_run:
                run_len += 1
            idx += 1
    if curr_run:
        run_len_compressed[-1][1] = run_len
    return run_len_compressed
