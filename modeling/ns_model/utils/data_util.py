# Modified from https://github.com/alexpashevich/E.T./blob/master/alfred/utils/data_util.py - Licensed under the
# MIT License.
#
# Modifications Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.


import os
import re
import json
import torch
import shutil
import pickle
import warnings
import numpy as np

from PIL import Image
from tqdm import tqdm
from io import BytesIO
from torch.nn.utils.rnn import pad_sequence
from torchvision import transforms
from copy import deepcopy
from pathlib import Path

from modeling.ns_model.utils.lang import indexesFromSentence

def read_images(image_path_list):
    images = []
    for image_path in image_path_list:
        image_orig = Image.open(image_path)
        images.append(image_orig.copy())
        image_orig.close()
    return images


def read_traj_images(json_path, image_folder):
    root_path = json_path.parents[0]
    with open(json_path) as json_file:
        json_dict = json.load(json_file)
    image_names = [None] * len(json_dict['plan']['low_actions'])
    for im_idx, im_dict in enumerate(json_dict['images']):
        if image_names[im_dict['low_idx']] is None:
            image_names[im_dict['low_idx']] = im_dict['image_name']
    before_last_image = json_dict['images'][-1]['image_name']
    last_image = '{:09d}.png'.format(int(before_last_image.split('.')[0]) + 1)
    image_names.append(last_image)
    fimages = [root_path / image_folder / im for im in image_names]
    if not any([os.path.exists(path) for path in fimages]):
        # maybe images were compressed to .jpg instead of .png
        fimages = [Path(str(path).replace('.png', '.jpg')) for path in fimages]
    if not all([os.path.exists(path) for path in fimages]):
        return None
    assert len(fimages) > 0
    # this reads on images (works with render_trajs.py)
    # fimages = sorted(glob.glob(os.path.join(root_path, image_folder, '*.png')))
    try:
        images = read_images(fimages)
    except:
        return None
    return images


def extract_features(images, extractor):
    if images is None:
        return None
    feat = extractor.featurize(images, batch=2)
    return feat.cpu()


def process_traj(traj_orig, traj_path, r_idx, preprocessor):
    # copy trajectory
    traj = traj_orig.copy()
    # root & split
    traj['root'] = str(traj_path)
    partition = traj_path.parents[2 if 'tests_' not in str(traj_path) else 1].name
    traj['split'] = partition
    traj['repeat_idx'] = r_idx
    # numericalize actions for train/valid splits
    if 'test' not in partition: # expert actions are not available for the test set
        preprocessor.process_actions(traj_orig, traj)
    # numericalize language
    preprocessor.process_language(traj_orig, traj, r_idx)
    return traj


def gather_jsons(files, output_path):
    print('Writing JSONs to PKL')
    if output_path.exists():
        os.remove(output_path)
    jsons = {}
    for idx, path in tqdm(enumerate(files)):
        with open(path, 'rb') as f:
            jsons_idx = pickle.load(f)
            jsons['{:06}'.format(idx).encode('ascii')] = jsons_idx
    with output_path.open('wb') as f:
        pickle.dump(jsons, f)

def tensorize_and_pad(batch, vocab_in, vocab_out, device, pad):
    '''
    cast values to torch tensors, put them to the correct device and pad sequences
    '''
    device = torch.device(device)
    input_dict, gt_dict, feat_dict = dict(), dict(), dict()
    feat_dict = batch
    
    # feat_dict keys that start with these substrings will be assigned to input_dict
    input_keys = {'lang', 'frames'}
    gt_keys = {'action', 'masks', 'object', 'mission_id', 'action_valid_interact'}
    for k, v in feat_dict.items():
        dict_assign = input_dict if any([k.startswith(s) for s in input_keys]) else gt_dict
        if k not in input_keys and k not in gt_keys:
            pass
        elif k.startswith('lang'):
            # no preprocessing should be done here
            seqs = [torch.tensor(indexesFromSentence(vocab_in, vv) if vv is not None else [pad, pad], device=device).long() for vv in v]
            pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
            dict_assign[k] = pad_seq
            dict_assign['lengths_' + k] = torch.tensor(list(map(len, seqs)))
            length_max_key = 'length_' + k + '_max'
            if ':' in k:
                # for translated length keys (e.g. lang:lmdb/1x_det) we should use different names
                length_max_key = 'length_' + k.split(':')[0] + '_max:' + ':'.join(k.split(':')[1:])
            dict_assign[length_max_key] = max(map(len, seqs))
        elif k in {'action'}:
            action_seqs = []
            for vv in v:
                seq = []
                for vvv in vv:
                    seq.append(vocab_out.word2index[vvv])
                action_seqs.append(torch.tensor(seq, device=device).long())

            pad_seq = pad_sequence(action_seqs, batch_first=True, padding_value=pad)
            dict_assign[k] = pad_seq
        elif k in {'object'}:
            obj_seqs = []
            for vv in v:
                seq = []
                for vvv in vv:
                    seq += indexesFromSentence(vocab_out, vvv)
                obj_seqs.append(torch.tensor(seq, device=device).long())

            dict_assign[k] = obj_seqs
        elif k in {'frames'}:
            # frames features 
            seqs = [vv.clone().detach().to(device).type(torch.float) for vv in v]
            pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
            dict_assign[k] = pad_seq
            dict_assign['lengths_' + k] = torch.tensor(list(map(len, seqs)))
            dict_assign['length_' + k + '_max'] = max(map(len, seqs))
        elif k in {'mission_id'}:
            seqs = v
            dict_assign[k] = seqs
        else:
            # default: tensorize and pad sequence
            seqs = [torch.tensor(vv, device=device, dtype=torch.long) for vv in v]
            pad_seq = pad_sequence(seqs, batch_first=True, padding_value=pad)
            dict_assign[k] = pad_seq
    return input_dict, gt_dict


def sample_batches(iterators, vocab_in, vocab_out, device, pad, args):
    '''
    sample a batch from each iterator, return Nones if the iterator is empty
    '''
    batches_dict = {}
    for dataset_id, iterator in iterators.items():
        try:
            batches = next(iterator)
        except StopIteration as e:
            return None
        dataset_name = dataset_id
        input_dict, gt_dict = tensorize_and_pad(
            batches, vocab_in, vocab_out, device, pad)
        batches_dict[dataset_name] = (input_dict, gt_dict)
    return batches_dict


def get_feat_shape(visual_archi, compress_type=None):
    '''
    Get feat shape depending on the training archi and compress type
    '''
    if visual_archi == 'fasterrcnn':
        # the RCNN model should be trained with min_size=224
        feat_shape = (-1, 2048, 7, 7)
    elif visual_archi == 'maskrcnn':
        # the RCNN model should be trained with min_size=800
        feat_shape = (-1, 2048, 10, 10)
    elif visual_archi == 'maskrcnn_v2':
         feat_shape = (-1, 2048, 25, 25)
    elif visual_archi == 'resnet18':
        feat_shape = (-1, 512, 7, 7)
    else:
        raise NotImplementedError('Unknown archi {}'.format(visual_archi))

    if compress_type is not None:
        if not re.match(r'\d+x', compress_type):
            raise NotImplementedError('Unknown compress type {}'.format(compress_type))
        compress_times = int(compress_type[:-1])
        feat_shape = (
            feat_shape[0], feat_shape[1] // compress_times,
            feat_shape[2], feat_shape[3])
    return feat_shape


def feat_compress(feat, compress_type):
    '''
    Compress features by channel average pooling
    '''
    assert re.match(r'\d+x', compress_type) and len(feat.shape) == 4
    times = int(compress_type[:-1])
    assert feat.shape[1] % times == 0
    feat = feat.reshape((
        feat.shape[0], times,
        feat.shape[1] // times,
        feat.shape[2],
        feat.shape[3]))
    feat = feat.mean(dim=1)
    return feat
