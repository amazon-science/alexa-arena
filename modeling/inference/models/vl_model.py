# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import os
import copy
import torch

from modeling.inference.util.robot_actions import *
from modeling.inference.util.utils import compress_mask
from modeling.inference.util.vl_model_utils import process_inputs_for_vl_model, postprocess_mask
from modeling.inference.util.name_mapping import valid_rooms


ML_TOOLBOX_BASE_DIR_PATH = os.getenv('ALEXA_ARENA_DIR')


def construct_request(act, obj, pred_mask):
    try:
        if (act not in OBJECT_INTERACT_ACTIONS):
            if (obj in valid_rooms):  # GoTo room action
                act = 'Goto'
                request = copy.deepcopy(GOTO_ROOM_ACTION)
                request[act.strip().lower()]["object"]["officeRoom"] = obj.strip()
            elif act == "Rotate":
                request = copy.deepcopy(ROTATE_ACTION)
                request["rotation"]["magnitude"] = 90.0
                if obj in {'Left', 'Right'}:
                    request["rotation"]["direction"] = obj.strip()
                else: # Rotate to an object -> go to the object
                    act = 'Goto'
                    request = copy.deepcopy(GOTO_OBJECT_ACTION)
                    compressed_mask = compress_mask(pred_mask)
                    request[act.strip().lower()]["object"]["mask"] = compressed_mask
            else:  # GoTo the mask
                request = copy.deepcopy(GOTO_OBJECT_ACTION)
                compressed_mask = compress_mask(pred_mask)
                request[act.strip().lower()]["object"]["mask"] = compressed_mask
        else:  # Interact
            request = copy.deepcopy(INTERACT_ACTION)
            compressed_mask = compress_mask(pred_mask)
            request["type"] = act.strip()
            request[act.strip().lower()] = request["dummy_act"]
            del request["dummy_act"]
            request[act.strip().lower()]["object"]["mask"] = compressed_mask
    except:
        print("Could not construct valid request from predicted action-object pair. "
              "Predictions: Action - {}, Object - {}".format(act, obj))
        # Dummy action to execute in case of an invalid action construction request
        request = {"id": "", "type": "Rotate",  "rotation": {"direction": "Right", "magnitude": 0.0}}
    return request

def predict_action_and_object_e2e_vl(vl_model, images_list, instruction, eos, ht_1, ct_1, id2act, id2obj, device):
    pixel_mask_prob_threshold = 0.35
    eos_threshold = 0.75
    for image in images_list:
        image_input, language_input = process_inputs_for_vl_model(image, instruction)
        image_input.to(device)
        language_input.to(device)
        if eos:
            ht_1, ct_1 = None, None
        pred_mask, action_logits_list, object_logits_list, eos_logits_list, ht, ct = vl_model(image_input, language_input, ht_1, ct_1)
        last_action = id2act[torch.argmax(action_logits_list[-1]).item()]
        last_object = id2obj[torch.argmax(object_logits_list[-1]).item()]
        last_eos = torch.sigmoid(eos_logits_list[-1]).item() > eos_threshold
        pred_mask = torch.sigmoid(pred_mask)
        pred_mask = postprocess_mask(pred_mask, image_input)
        pred_mask[pred_mask >= pixel_mask_prob_threshold] = 1
        pred_mask[pred_mask < pixel_mask_prob_threshold] = 0
        request = construct_request(last_action, last_object, pred_mask)
    return request, last_eos, ht, ct
