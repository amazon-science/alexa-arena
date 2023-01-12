# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import os

import boto3
from scipy.spatial.distance import cosine
import torch
from torchvision.transforms import functional as F
from modeling.vl_model.data_generators.vl_data_generator import tokenize
import torch.nn.functional as I
import numpy as np
from modeling.vl_model.data_generators.vl_data_generator import tokenize
import cv2
import modeling.inference.constants as constants


def compute_sentence_embedding(text, bert_model, tokenizer):
    marked_text = "[CLS] " + text + " [SEP]"

    # Tokenize our sentence with the BERT tokenizer.
    tokenized_text = tokenizer.tokenize(marked_text)

    # Map the token strings to their vocabulary indeces.
    indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)

    segments_ids = [1] * len(tokenized_text)

    tokens_tensor = torch.tensor([indexed_tokens])
    segments_tensors = torch.tensor([segments_ids])

    with torch.no_grad():
        encoded_layers, _ = bert_model(tokens_tensor, segments_tensors)
    token_embeddings = torch.stack(encoded_layers, dim=0)
    token_vecs = encoded_layers[11][0]

    # Calculate the average of all 22 token vectors.
    sentence_embedding = torch.mean(token_vecs, dim=0)

    return sentence_embedding


def compute_similarity(sentence_embedding_1, sentence_embedding_2):
    if type(sentence_embedding_1) == torch.Tensor:
        sentence_embedding_1 = sentence_embedding_1.cpu().detach().numpy()
    if type(sentence_embedding_2) == torch.Tensor:
        sentence_embedding_2 = sentence_embedding_2.cpu().detach().numpy()
    return 1 - cosine(sentence_embedding_1, sentence_embedding_2)


def restore_checkpoint(path_to_model_checkpoint, model, device, optimizer=None):
    print(f"Restoring checkpoint {path_to_model_checkpoint}")
    checkpoint = torch.load(path_to_model_checkpoint,  map_location=torch.device(device))
    state_dict = checkpoint["model"]
    try:
        new_state_dict = make_state_dict_compatible(state_dict)
        model.model.load_state_dict(new_state_dict)
    except RuntimeError:
        state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
        new_state_dict = make_state_dict_compatible(state_dict)
        model.model.load_state_dict(new_state_dict)
    if optimizer:
        optimizer.load_state_dict(checkpoint['optim'])
    return model, optimizer

def make_state_dict_compatible(state_dict):
    state_dict_new2old_map = {
        "backbone.fpn.inner_blocks.0.0.weight":  "backbone.fpn.inner_blocks.0.weight",
        "backbone.fpn.inner_blocks.0.0.bias": "backbone.fpn.inner_blocks.0.bias",
        "backbone.fpn.inner_blocks.1.0.weight": "backbone.fpn.inner_blocks.1.weight",
        "backbone.fpn.inner_blocks.1.0.bias": "backbone.fpn.inner_blocks.1.bias",
        "backbone.fpn.inner_blocks.2.0.weight": "backbone.fpn.inner_blocks.2.weight",
        "backbone.fpn.inner_blocks.2.0.bias": "backbone.fpn.inner_blocks.2.bias",
        "backbone.fpn.inner_blocks.3.0.weight":  "backbone.fpn.inner_blocks.3.weight",
        "backbone.fpn.inner_blocks.3.0.bias": "backbone.fpn.inner_blocks.3.bias",
        "backbone.fpn.layer_blocks.0.0.weight": "backbone.fpn.layer_blocks.0.weight",
        "backbone.fpn.layer_blocks.0.0.bias": "backbone.fpn.layer_blocks.0.bias",
        "backbone.fpn.layer_blocks.1.0.weight": "backbone.fpn.layer_blocks.1.weight",
        "backbone.fpn.layer_blocks.1.0.bias": "backbone.fpn.layer_blocks.1.bias",
        "backbone.fpn.layer_blocks.2.0.weight": "backbone.fpn.layer_blocks.2.weight",
        "backbone.fpn.layer_blocks.2.0.bias":  "backbone.fpn.layer_blocks.2.bias",
        "backbone.fpn.layer_blocks.3.0.weight": "backbone.fpn.layer_blocks.3.weight",
        "backbone.fpn.layer_blocks.3.0.bias": "backbone.fpn.layer_blocks.3.bias",
        "rpn.head.conv.0.0.weight":  "rpn.head.conv.weight",
        "rpn.head.conv.0.0.bias": "rpn.head.conv.bias",
        "roi_heads.mask_head.0.0.weight": "roi_heads.mask_head.mask_fcn1.weight",
        "roi_heads.mask_head.0.0.bias": "roi_heads.mask_head.mask_fcn1.bias",
        "roi_heads.mask_head.1.0.weight": "roi_heads.mask_head.mask_fcn2.weight",
        "roi_heads.mask_head.1.0.bias": "roi_heads.mask_head.mask_fcn2.bias",
        "roi_heads.mask_head.2.0.weight": "roi_heads.mask_head.mask_fcn3.weight",
        "roi_heads.mask_head.2.0.bias": "roi_heads.mask_head.mask_fcn3.bias",
        "roi_heads.mask_head.3.0.weight": "roi_heads.mask_head.mask_fcn4.weight",
        "roi_heads.mask_head.3.0.bias": "roi_heads.mask_head.mask_fcn4.bias"
    }
    new_state_dict = {}
    for key in state_dict:
        if key in state_dict_new2old_map:
            new_state_dict[state_dict_new2old_map[key]] = state_dict[key]
        else:
            new_state_dict[key] = state_dict[key]
    return new_state_dict

def read_file_from_s3(s3_client, bucket, key):
    """
    Read file from s3 bucket.
    """
    s3_object = s3_client.get_object(Bucket=bucket, Key=key)
    data = s3_object['Body'].read().decode()
    return data


def write_file_to_s3(s3_client, byte_data, bucket_name, key_name):
    """
    Write file to s3 bucket.
    """
    s3_client.put_object(Bucket=bucket_name, Key=key_name, Body=byte_data)


def upload_file_to_s3(s3_client, local_fn, dst_bucket, dst_path):
    """
    Upload file to s3 bucket.
    """
    s3_client.upload_file(local_fn, dst_bucket, dst_path)


def upload_folder_to_s3(s3_client, src_folder, dst_bucket, dst_folder):
    """
    Upload all files in a folder to bucket.
    """
    # list files.
    src_fns = os.listdir(src_folder)
    for src_fn in src_fns:
        upload_file_to_s3(s3_client, os.path.join(src_folder, src_fn),
                          dst_bucket, dst_folder + "/" + src_fn)


class AWSHelper(object):
    def __init__(self):
        self.session = boto3.session.Session()

    def get_s3(self):
        s3_client = self.session.client('s3')
        return s3_client

aws_helper = AWSHelper()



################################################################################
# instance segmentation utility functions
################################################################################
def compress_mask(seg_mask):
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

def process_image_for_model(image):
    image = image.copy()
    image = F.to_tensor(image)
    return tuple([image])

def process_inputs_for_vl_model(image, utterance='', oracle_output=None):
    mean = torch.tensor(constants.means).reshape(3, 1, 1)
    std = torch.tensor(constants.stds).reshape(3, 1, 1)
    image = torch.from_numpy(image.transpose((2, 0, 1)))
    if not isinstance(image, torch.FloatTensor):
        image = image.float()
        image.div_(255.).sub_(mean).div_(std)
    sentences = [utterance.strip()]
    if oracle_output:
        sentences.append(oracle_output['text'].strip().lower())
    language_input = [' '.join(sentences)]
    print(sentences)
    text = tokenize(language_input, 22, True)
    image = image.cuda(non_blocking=True)
    text = text.cuda(non_blocking=True)
    return torch.unsqueeze(image, 0), text

def postprocess_mask(pred, img):
    if pred.shape[-2:] != img.shape[-2:]:
        pred = I.interpolate(
            pred,
            size=img.shape[-2:],
            mode='bicubic',
            align_corners=True).squeeze()
    mat = np.array([[1., 0., 0.],
                    [0., 1., 0.]])
    w, h = 300, 300
    pred = pred.cpu().numpy()
    pred = cv2.warpAffine(pred, mat, (w, h),
                          flags=cv2.INTER_CUBIC,
                          borderValue=0.)
    return pred