# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1

import json
import os
from collections import defaultdict
import numpy as np
import torch
from torch.utils.data import Dataset
import copy
import pickle
from numpy.random import default_rng

from modeling.ns_model.utils.utils import normalizeString
from modeling.ns_model.utils.lang import Lang
from modeling.ns_model.utils.data_util import read_images, extract_features
from modeling.ns_model.nn.enc_visual import FeatureExtractor

class ArenaNSDataset(Dataset):
    def __init__(self, args, metadata_file, images_root, split, annotation_type):
        self.args = args
        self.metadata_file = metadata_file
        self.images_root = images_root
        self.split = split
        self.name = "human"
        self.data = []
        self.rng = default_rng(12345)
        self.ann_type = annotation_type

        with open(self.args.data_dir + "vision-data/obj_id_to_class_customized.json", "r") as f:
            self.id2class = json.load(f)

        self.vocab_in = Lang(self.split + ":" + self.name)
        self.vocab_out = Lang(self.split + ":" + self.name)
        self.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.visual_archi = self.args.visual_archi

        # use fasterrcnn pretrained on coco
        if self.visual_archi == "fasterrcnn":
            self.extractor = FeatureExtractor(self.visual_archi, device=self.device, checkpoint=None, share_memory=True)
        elif self.visual_archi == "maskrcnn_v2":
            self.extractor = FeatureExtractor(self.visual_archi, device=self.device, \
                checkpoint=self.args.vision_model_path, share_memory=True)
        else:
            raise Exception("visual architecture %s not supported" % self.visual_archi)

        instruction_action_object_pairs = self.build_dataset()
        self.prep_vocab(instruction_action_object_pairs)

    def build_dataset(self):
        with open(self.metadata_file) as f:
            train_data = json.load(f)

        mission_count = 0
        print("Building NS dataset from {}".format(self.metadata_file))
        unassigned_object_id = []
        pairs = []
        for mission_id in train_data:
            mission_count += 1
            # print mission count every 500 data points
            if (mission_count % 500) == 0:
                print('Prepared {} instances'.format(mission_count))
            if self.ann_type == "h":
                human_annotations = train_data[mission_id]['human_annotations']
            elif self.ann_type == "s":
                human_annotations = train_data[mission_id]['synthetic_annotations']
            else:
                raise Exception("Annotation type %s not defined" % self.ann_type)

            path_to_mission_images = os.path.join(self.images_root, mission_id)

            all_actions = train_data[mission_id]["actions"]
            for human_ann in human_annotations:
                norm_instr_seq = ""
                action_seq = []
                obj_seq = []
                color_image_files = []
                all_action_idx = []

                for instr_qa in human_ann['instructions']:
                    # add questions and answers together with the instruction
                    instruction = " <<instr>> " + instr_qa['instruction']
                    instruction_qa = copy.deepcopy(instruction)
                    qas = None
                    if 'question_answers' in instr_qa:
                        qas = instr_qa['question_answers']
                        for qa in qas:
                            qst = qa["question"].lower()
                            qtype, q_obj = self.prep_question_tokens(qst)
                            ans = qa["answer"]
                            instruction_qa += qtype + q_obj + " <<ans>> " + ans

                    actions_idx = instr_qa["actions"]
                    all_action_idx.append(actions_idx)
                    actions = [all_actions[idx] for idx in actions_idx]

                    for act in actions:
                        a_type = act["type"].lower()
                        # prepare the image paths
                        color_image_idx = None
                        if 'object' in act[a_type]:
                            color_image_idx = act[a_type].get('object').get('colorImageIndex')
                        color_image_file = None
                        # append images to the list for rotation
                        if color_image_idx is not None and color_image_idx > 0:
                            color_image_files += [os.path.join(path_to_mission_images, act['colorImages'][c]) for c in range(color_image_idx+1)]
                        else:
                            color_image_file = os.path.join(path_to_mission_images, act['colorImages'][0])
                            color_image_files.append(color_image_file)
                        
                        # prepare the action and object label
                        o_word = self.prep_obj_label(act, a_type)
                        a_word = a_type

                        # add additional actions and objects for rotation
                        if color_image_idx is not None and color_image_idx > 0:
                            action_seq += ["rotate" for c in range(color_image_idx)]
                            obj_seq += ["Unassigned" for c in range(color_image_idx)]

                        action_seq.append(a_word)
                        obj_seq.append(o_word.replace(" ", "_"))

                    norm_instr = normalizeString(instruction_qa)
                    norm_instr_seq += " " + norm_instr
                    pairs.append((norm_instr, action_seq, obj_seq))

                action_seq.append("<stop>")
                obj_seq.append("Unassigned")
                # add last image to the image list
                last_act_img_path = color_image_files[-1]
                path_splits = last_act_img_path.split("-action-num-")
                last_act_img_idx = int(path_splits[1].split("_colorImage_")[0]) + 1
                last_img_path = path_splits[0] + "-action-num-" + str(last_act_img_idx) + "_colorImage_" + (path_splits[1].split("_colorImage_")[1])
                color_image_files.append(last_img_path)

                self.data.append({
                    'lang': norm_instr_seq,
                    'action': action_seq,
                    'object': obj_seq, 
                    'action_idx': all_action_idx, 
                    'color_image_files': color_image_files,
                    'mission_id': mission_id,
                    'action_valid_interact': [1 for j in range(len(action_seq))], 
                })
        
        return pairs

    def prep_vocab(self, pairs):
        # add instructions, actions and objects to the vocabulary
        self.vocab_out.addSentence("<stop>")
        for pair in pairs:
            self.vocab_in.addSentence(pair[0])
            for p in pair[1]:
                self.vocab_out.addSentence(p)
            for p in pair[2]:
                self.vocab_out.addSentence(p)

    @staticmethod
    def prep_question_tokens(qst):
        # prepare the question tokens
        if "where is" in qst:
            # for predefined questions, use <<qtype>> + <<q_obj>> as the tokens
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
            # for free-form questions, use <<q>> + the whole sentence as the tokens
            qtype = " <<q>> "
            q_obj = qst
        
        return qtype, q_obj

    def prep_obj_label(self, act, a_type):
        # prepare the object label use vision class labels
        if act["type"] == "Look":
            o_word = "Unassigned"
        elif "id" in act[a_type]["object"]:
            o_id = act[a_type]["object"]["id"]
            o_id_trim = "_".join(o_id.split("_")[:-1])
            if o_id_trim not in self.id2class:
                raise Exception("object id %s cannot be recognized by the vision model" % o_id_trim)
            o_word = self.id2class[o_id_trim]
            # some objects have unassigned class labels
            if o_word == "Unassigned":
                pass
        elif "officeRoom" in act[a_type]["object"]:
            o_word = act[a_type]["object"]["officeRoom"]
        else:
            raise Exception("object id not detected in the data")
        
        return o_word


    def __getitem__(self, index):
        data_orig = copy.deepcopy(self.data[index])
        path_to_imgs = data_orig['color_image_files']
        imgs = read_images(path_to_imgs)
        feat = extract_features(imgs, self.extractor)
        data_orig['frames'] = feat.to('cpu')
        del data_orig['color_image_files']
        del feat
        
        return data_orig

    def __len__(self):
        return len(self.data)

    def save_vocab_in(self):
        model_path = self.args.checkpt_dir+str(self.args.exp_num)+"/"
        with open(model_path+"%s_%s_vocabin.pkl" % (self.ann_type, self.split), "wb") as f:
            pickle.dump(self.vocab_in, f)

