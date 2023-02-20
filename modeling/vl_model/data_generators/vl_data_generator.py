# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import json
import os
import sys
from collections import defaultdict

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset
import cv2
import random
from typing import List, Union
from string import punctuation

sys.path.append(f"{os.environ['HOME']}/AlexaArena/")
from modeling.vl_model.data_generators.simple_tokenizer import SimpleTokenizer as _Tokenizer
from arena_wrapper.util import (
    decompress_mask,
    DETECTION_SCREEN_WIDTH, DETECTION_SCREEN_HEIGHT
)


_tokenizer = _Tokenizer()

def tokenize(texts: Union[str, List[str]],
             context_length: int = 77,
             truncate: bool = False) -> torch.LongTensor:
    """
    Returns the tokenized representation of given input string(s)
    Params:
        texts : Union[str, List[str]]
            An input string or a list of input strings to tokenize

        context_length : int
            The context length to use; all CLIP models use 77 as the context length

        truncate: bool
            Whether to truncate the text in case its encoding is longer than the context length
    Returns:
        A two-dimensional tensor containing the resulting tokens,
        shape = [number of input strings, context_length]
    """
    if isinstance(texts, str):
        texts = [texts]

    sot_token = _tokenizer.encoder["<|startoftext|>"]
    eot_token = _tokenizer.encoder["<|endoftext|>"]
    all_tokens = [[sot_token] + _tokenizer.encode(text) + [eot_token]
                  for text in texts]
    result = torch.zeros(len(all_tokens), context_length, dtype=torch.long)

    for i, tokens in enumerate(all_tokens):
        if len(tokens) > context_length:
            if truncate:
                tokens = tokens[:context_length]
                tokens[-1] = eot_token
            else:
                raise RuntimeError(
                    f"Input {texts[i]} is too long for context length {context_length}"
                )
        result[i, :len(tokens)] = torch.tensor(tokens)

    return result

class ArenaRefDataset(Dataset):
    def __init__(self, metadata_file, images_root, split, args):
        self.metadata_file = metadata_file
        self.images_root = images_root
        self.split = split
        self.data = []
        self.input_size = (300, 300)
        self.word_length = args.word_len
        self.mean = torch.tensor([0.48145466, 0.4578275,
                                  0.40821073]).reshape(3, 1, 1)
        self.std = torch.tensor([0.26862954, 0.26130258,
                                 0.27577711]).reshape(3, 1, 1)
        self.build_referential_dataset()
        id2class_path = args.obj_id_to_class_file
        class2idx_path = args.class_to_idx_file
        with open(class2idx_path) as f:
            self.class2idx = json.load(f)
        room_objs = {
            'BreakRoom': 86,
            'Reception': 87,
            'Lab2': 88,
            'Lab1': 89,
            'SmallOffice': 90,
            'MainOffice': 91,
            'Left': 92,
            'Right': 93
        }
        self.class2idx.update(room_objs)
        with open(id2class_path) as f:
            self.id2class = json.load(f)
        self.act2idx = {
            'Pickup': 0,
            'Break': 1,
            'Close': 2,
            'Open': 3,
            'Pour': 4,
            'Scan': 5,
            'Goto': 6,
            'Place': 7,
            'Toggle': 8,
            'Clean': 9,
            'Fill': 10,
            'Rotate': 11
        }

    def read_image(self, path):
        im_frame = Image.open(path)
        np_frame = np.array(im_frame)[:,:,0:3]  # Leaving out the transperancy channel
        return np_frame

    def build_referential_dataset(self):
        with open(self.metadata_file) as f:
            train_data = json.load(f)
        i = 0
        print("Building referential dataset from {}".format(self.metadata_file))
        self.mission_id_to_action_id_to_action_dict = defaultdict(dict)
        for mission_id in train_data:
            i += 1
            if (i % 500) == 0:
                print('Prepared {} mission ids'.format(i))
            human_annotations = train_data[mission_id]['human_annotations']
            actionid_to_language_dict = defaultdict(list)
            language_prompt_id_to_action_ids = defaultdict(list)
            for ann_id, human_ann in enumerate(human_annotations):
                for instr_id, ann_dict in enumerate(human_ann['instructions']):
                    instruction = ann_dict.get('instruction')
                    action_ids = ann_dict.get('actions')
                    question_answers = ann_dict.get('question_answers') if ann_dict.get('question_answers') != None else []
                    language_prompt_id_to_action_ids['{}_instr{}_ann{}'.format(mission_id, instr_id, ann_id)] = action_ids
                    for action_id in action_ids:
                        actionid_to_language_dict[action_id].append(
                            {'instruction': instruction,
                            'question_answers': question_answers,
                            'mission_id_instr_id_ann_id': '{}_instr{}_ann{}'.format(mission_id, instr_id, ann_id)})
            path_to_mission_images = os.path.join(self.images_root, mission_id)
            for action in train_data[mission_id]['actions']:
                action_type = action['type']
                action_dict = action[action_type.lower()]
                object_instance = action_dict.get('object')
                action_id = action.get('id')
                self.mission_id_to_action_id_to_action_dict[mission_id][action_id] = action
                if object_instance != None:  # for non-look around commands
                    language_prompts = actionid_to_language_dict[action_id]
                    compressed_mask = object_instance.get('mask')
                    # If the "object" is not a room or viewpoint, which does not require a mask
                    if compressed_mask != None:
                        mask = decompress_mask(compressed_mask)
                        color_image_idx = object_instance.get('colorImageIndex')
                        color_image_file = os.path.join(
                            path_to_mission_images,
                            action['colorImages'][color_image_idx])
                        instance_segmentation_file = os.path.join(
                                path_to_mission_images,
                                action['colorImages'][color_image_idx].replace('colorImage', 'instanceSegmentationImage'))
                        objects_payload_file = os.path.join(
                                path_to_mission_images,
                                '_'.join(action['colorImages'][color_image_idx].split('_')[0:-2]) + '_objects_payload.json')
                        for lp in language_prompts:
                            self.data.append({
                                'action': action,
                                'mask': mask,
                                'color_image_file': color_image_file,
                                'language_prompt': lp,
                                'action_ids': language_prompt_id_to_action_ids[lp['mission_id_instr_id_ann_id']],
                                'mission_id': mission_id,
                                'instance_segmentation_file': instance_segmentation_file,
                                'objects_payload_file': objects_payload_file})
                    else:  # if the object is go to a room or viewpoint which doesn't need a mask
                        mask = np.zeros((DETECTION_SCREEN_WIDTH, DETECTION_SCREEN_HEIGHT))
                        for color_image_file in action['colorImages']:
                            color_image_file = os.path.join(
                                path_to_mission_images,
                                color_image_file)
                            instance_segmentation_file = os.path.join(
                                path_to_mission_images,
                                color_image_file.replace('colorImage', 'instanceSegmentationImage'))
                            objects_payload_file = os.path.join(
                                path_to_mission_images,
                                '_'.join(color_image_file.split('_')[0:-2]) + '_objects_payload.json')
                            for lp in language_prompts:
                                self.data.append({
                                        'action': action,
                                        'mask': mask,
                                        'color_image_file': color_image_file,
                                        'language_prompt': lp,
                                        'action_ids': language_prompt_id_to_action_ids[lp['mission_id_instr_id_ann_id']],
                                        'mission_id': mission_id,
                                        'instance_segmentation_file': instance_segmentation_file,
                                        'objects_payload_file': objects_payload_file}) 
                else:
                    # ensure that all other non mask requiring actions are look around commands, which we don't include
                    # because the user cannot invoke this command using language, its usage is limited to the model
                    # the resulting images of the look around command are actually incorporated into the subsequent commands
                    # in the action sequence.
                    assert(action_type == "Look")

    def convert(self, img, mask=None):
        # Image ToTensor & Normalize
        img = torch.from_numpy(img.transpose((2, 0, 1)))
        if not isinstance(img, torch.FloatTensor):
            img = img.float()
        img.div_(255.).sub_(self.mean).div_(self.std)
        # Mask ToTensor
        if mask is not None:
            mask = torch.from_numpy(mask)
            if not isinstance(mask, torch.FloatTensor):
                mask = mask.float()
        return img, mask

    def getTransformMat(self, img_size, inverse=False):
        ori_h, ori_w = img_size
        inp_h, inp_w = self.input_size
        scale = min(inp_h / ori_h, inp_w / ori_w)
        new_h, new_w = ori_h * scale, ori_w * scale
        bias_x, bias_y = (inp_w - new_w) / 2., (inp_h - new_h) / 2.

        src = np.array([[0, 0], [ori_w, 0], [0, ori_h]], np.float32)
        dst = np.array([[bias_x, bias_y], [new_w + bias_x, bias_y],
                        [bias_x, new_h + bias_y]], np.float32)

        mat = cv2.getAffineTransform(src, dst)
        if inverse:
            mat_inv = cv2.getAffineTransform(dst, src)
            return mat, mat_inv
        return mat, None

    def read_and_process_image_and_mask(self, path_to_img, mask):
        img = self.read_image(path_to_img)
        ori_img = img
        img_size = img.shape[:2]
        mat, mat_inv = self.getTransformMat(img_size, True)
        img = cv2.warpAffine(
            img,
            mat,
            self.input_size,
            flags=cv2.INTER_CUBIC,
            borderValue=[0.48145466 * 255, 0.4578275 * 255, 0.40821073 * 255])
        mask = cv2.warpAffine(mask,
                            mat,
                            self.input_size,
                            flags=cv2.INTER_LINEAR,
                            borderValue=0.)
        img, mask = self.convert(img, mask)
        return img, mat, mat_inv, mask, ori_img

    def get_question_type(self, qst):
        qst = qst.lower()
        if "where is" in qst:
            qtype = " <<loc>> "
            q_obj = qst.split()[-1][:-1]
        elif "look like" in qst:
            qtype = " <<app>> "
            q_obj = qst.split()[3]
        elif "referring to" in qst:
            qtype = " <<ref>> "
            q_obj = qst.split()[1]
        elif "which direction" in qst:
            qtype = " <<dir>> "
            q_obj = ""
        else:
            # include the whole sentence for free-form questions
            qtype = " <<q>> "
            q_obj = qst
        return (qtype, q_obj)



    def get_action_obect_idxs(self, action_dict):
        action_id = action_dict["type"]
        action_idx = self.act2idx.get(action_id)
        if action_idx is None:
            action_idx = 0
        object_instance = action_dict[action_dict["type"].lower()].get('object')
        object_id = object_instance.get('id')
        room_instance = object_instance.get('officeRoom')
        class_key = 'Unassigned'
        if object_id is not None:
            class_key = self.id2class["_".join(object_id.split("_")[:-1])]
        if room_instance is not None:
            class_key = room_instance.strip()
        object_idx = self.class2idx[class_key]
        return [action_idx, object_idx, 0]  # The last element indicates whether it is the last in the sequence

    def __getitem__(self, index):
        data = self.data[index]
        action_dicts = [self.mission_id_to_action_id_to_action_dict[data['mission_id']][aid]
                        for aid in data['action_ids']]
        mission_id = data['mission_id']
        path_to_mission_images = os.path.join(self.images_root, mission_id) 
        pruned_images, pruned_masks, action_object_pairs, ori_pruned_images = [], [], [], []
        for action_dict in action_dicts:
            if not action_dict["type"] == "Look":
                object_instance = action_dict[action_dict["type"].lower()].get('object')
                if object_instance != None:  # for non-look around commands
                    compressed_mask = object_instance.get('mask')
                    color_image_idx = object_instance.get('colorImageIndex')
                    if color_image_idx is None:
                        color_image_idx = len(action_dict['colorImages']) - 1
                        assert(color_image_idx==0)
                    # Adding turnright and left commands for choosing 
                    # look around images during final execution
                    # the angle is always assumed to be 90
                    if color_image_idx == 0:
                        pass
                    elif color_image_idx == 1:
                        color_image_file = os.path.join(
                            path_to_mission_images,
                            action_dict['colorImages'][0])
                        mask = np.zeros((DETECTION_SCREEN_WIDTH,
                                         DETECTION_SCREEN_HEIGHT))
                        img, mat, mat_inv, mask, ori_img  = \
                            self.read_and_process_image_and_mask(color_image_file, mask)
                        pruned_images.append(img)
                        pruned_masks.append(mask)
                        ori_pruned_images.append(ori_img)
                        action_object_pairs.append([11, 93, 0])
                    elif color_image_idx == 2:
                        color_image_file = os.path.join(
                            path_to_mission_images,
                            action_dict['colorImages'][0])
                        mask = np.zeros((DETECTION_SCREEN_WIDTH, DETECTION_SCREEN_HEIGHT))
                        img, mat, mat_inv, mask, ori_img = \
                            self.read_and_process_image_and_mask(color_image_file, mask)
                        pruned_images.append(img)
                        pruned_masks.append(mask)
                        ori_pruned_images.append(ori_img)
                        # Randomly choosing either two lefts or two rights if it's directly behind
                        if np.random.uniform(low=0.0, high=1.0) < 0.5:
                            _idx = 3
                            action_object_pairs.extend([[11, 92, 0], [11, 92, 0]])
                        else:
                            _idx = 1
                            action_object_pairs.extend([[11, 93, 0], [11, 93, 0]]) 
                        mask = np.zeros((DETECTION_SCREEN_WIDTH, 
                                         DETECTION_SCREEN_HEIGHT))
                        color_image_file = os.path.join(
                            path_to_mission_images,
                            action_dict['colorImages'][_idx])
                        img, mat, mat_inv, mask, ori_img = \
                            self.read_and_process_image_and_mask(color_image_file, mask)
                        pruned_images.append(img)
                        pruned_masks.append(mask)
                        ori_pruned_images.append(ori_img)
                    elif color_image_idx == 3:
                        color_image_file = os.path.join(
                            path_to_mission_images,
                            action_dict['colorImages'][0])
                        mask = np.zeros((DETECTION_SCREEN_WIDTH,
                                         DETECTION_SCREEN_HEIGHT))
                        img, mat, mat_inv, mask, ori_img = \
                            self.read_and_process_image_and_mask(color_image_file, mask)
                        pruned_images.append(img)
                        pruned_masks.append(mask)
                        ori_pruned_images.append(ori_img)
                        action_object_pairs.extend([[11, 92, 0]])
                    color_image_file = os.path.join(
                        path_to_mission_images,
                        action_dict['colorImages'][color_image_idx])
                    # If the "object" is not a room or viewpoint, which does not require a mask
                    if compressed_mask != None:
                        mask = decompress_mask(compressed_mask)
                    else:
                        mask = np.zeros((DETECTION_SCREEN_WIDTH,
                                         DETECTION_SCREEN_HEIGHT))
                    img, mat, mat_inv, mask, ori_img = \
                        self.read_and_process_image_and_mask(color_image_file, mask)
                    pruned_masks.append(mask)
                    ori_pruned_images.append(ori_img)
                    # pruned_actions.append(action_dict)
                    pruned_images.append(img)
                    action_object_pairs.append(self.get_action_obect_idxs(action_dict))

        # Updating the last sequence with an EOS indicator
        action_object_pairs[-1][-1] = 1
        ori_img = img
        mask = data['mask']
        img_size = img.shape[:2]
        seg_id = '-1'
        try:
            answers = [qa['answer'] for qa in data['language_prompt']['question_answers']]
            questions = [qa['question'] for qa in data['language_prompt']['question_answers']]
            qtype_objs = [self.get_question_type(qst) for qst in questions]
            random_index = random.sample([x for x in range(len(answers))], k=1)[0]
            answers = [answers[random_index]]
            qtypes = [qtype_objs[random_index][0]]
        except:
            answers = []
            questions = []
            qtype_objs = []
            qtypes = []
        instructions = [data['language_prompt']['instruction']]
        instructions.extend(qtypes)
        instructions.extend(answers)
        sents = [instructions]
        sentence_list = [[s.strip(punctuation) for s in sents[0]]]
        sents = [[' '.join(sentence_list[0])]]
        if self.split == 'train':
            sent = random.choice(sents)
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            assert (len(pruned_masks) == len(pruned_images) == len(action_object_pairs))
            sequential_data = {
                'action_dicts': action_dicts,
                'sent': sent,
                'masks': pruned_masks,
                'ori_img': ori_pruned_images,
                'action_object_pairs': action_object_pairs,
                'images': pruned_images,
                }
            return word_vec, sequential_data
        elif self.split == 'validation':
            # sentence -> vector
            sent = sents[0]
            word_vec = tokenize(sent, self.word_length, True).squeeze(0)
            img = self.convert(img)[0]
            params = {
                'mask_dir': '',
                'inverse': mat_inv,
                'mask': mask,
                'ori_size': np.array(img_size),
                'sent': sent,
            }
            return img, word_vec, params
        else:
            # sentence -> vector
            img = self.convert(img)[0]
            params = {
                'ori_img': ori_img,
                'seg_id': seg_id,
                'mask_dir': '',
                'mask': mask,
                'inverse': mat_inv,
                'ori_size': np.array(img_size),
                'sents': sents[0],
            }
            return img, params

    def __len__(self):
        return len(self.data)
