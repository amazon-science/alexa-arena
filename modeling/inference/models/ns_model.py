# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import os
import sys
import re
import uuid
import json
import copy
import numpy as np
import torch

from modeling.inference.util.utils import compress_mask, process_image_for_model
from modeling.inference.util.robot_actions import ROTATE_ACTION, LOOK_AROUND_ACTION
from modeling.inference.util.name_mapping import valid_rooms, object_to_vision_class
from modeling.ns_model.utils import model_util

ALEXA_ARENA_DIR = os.getenv('ALEXA_ARENA_DIR')

def detect_uttered_color(instruction):
    instance_color = 'none'
    if "red" in instruction:
        instance_color = "red"
    elif "green" in instruction:
        instance_color = "green"
    elif "blue" in instruction:
        instance_color = "blue"
    return instance_color


def detect_instance_color(masked_instance, uttered_color):
    mask_color = 'none'
    idx_to_color = {
        0: "red",
        1: "green",
        2: "blue"
    }
    if uttered_color != 'none':
        r_channel, g_channel, b_channel  = masked_instance[:,:,0], masked_instance[:,:,1], masked_instance[:,:,2]
        r_g_b  = [np.sum(r_channel[r_channel != 0]), np.sum(g_channel[g_channel != 0]), np.sum(b_channel[b_channel != 0])]
        max_color_idx = np.argmax(r_g_b)
        mask_color = idx_to_color[max_color_idx]
    return mask_color


# Processes the outputs of the CV model to return a sings matching instance mask prediction corresponding to the object instance ID.
def process_outputs(outputs_list: dict, images_list: list, object_class: str, instruction: str=''):
    class_to_idx_path = os.path.join(ALEXA_ARENA_DIR, "data/vision-data/class_to_idx.json")
    with open(class_to_idx_path) as f:
        class2idx = json.load(f)
    idx2class = {v: k for k, v in class2idx.items()}
    predicted_class, predicted_score, predicted_mask = None, None, None

    score_threshold = 0.0
    img_idx = -1
    mask_sum = -1.0
    pixel_mask_prob_threshold = 0.75
    # maps semantic parser model classes to vision model classes
    if object_class in object_to_vision_class:
        class_of_object_id = object_to_vision_class[object_class]
    else:
        class_of_object_id = object_class
    uttered_instance_color = detect_uttered_color(instruction)
    for i, outputs in enumerate(outputs_list):
        pred_masks = outputs[0]["masks"].cpu()
        pred_labels = outputs[0]["labels"].cpu()
        scores = outputs[0]["scores"].cpu()
        for m in range(pred_masks.shape[0]):
            pred_score = scores[m].item()
            if pred_score > score_threshold:
                pred_label = idx2class[pred_labels[m].item()]
                if 'shelf' in pred_label.lower():
                    pred_label = 'Shelf'
                print('Recognized object: ' + pred_label + '  ' + str(pred_score))
                pred_mask = pred_masks[m][0].cpu().detach().numpy()
                pred_mask[pred_mask >= pixel_mask_prob_threshold] = 1
                pred_mask[pred_mask < pixel_mask_prob_threshold] = 0
                # take confidence score into consideration
                curr_mask_sum = np.sum(pred_mask) * pred_score
                masked_instance = images_list[i] * pred_mask[:, :, None]
                instance_color = detect_instance_color(masked_instance, uttered_instance_color)
                # if there are multiple instances of the object, we return the instance with the maximum score
                if (pred_label == class_of_object_id):
                    if (detect_instance_color(masked_instance, uttered_instance_color) == uttered_instance_color):
                        if (curr_mask_sum  > mask_sum):
                            mask_sum = curr_mask_sum
                            predicted_class = pred_label
                            predicted_mask = pred_mask
                            predicted_score = pred_score
                            img_idx = i
    return predicted_class, predicted_mask, img_idx


# Gets object name from command, maps the object name to a class instance segmentation mask 
# from the CV model and adds the binary mask predicted by the model for that instance of the object.
def process_gt_mask(command, images_list, device, cpu_device, model, instruction=''):
    # some commands do not require object grounding
    if command["type"].lower() == 'rotate' or "object" not in command[command["type"].lower()]:
        return command
    if command["type"].lower() == 'goto' and any(x in command['goto']["object"] for x in ("goToPoint", "officeRoom")):
        return command

    # get object name
    if "name" in command[command["type"].lower()]["object"]:
        object_class = command[command["type"].lower()]["object"]["name"]
    else:
        print("Object name cannot be determined!")
        return command

    outputs_list = []
    for image in images_list:
        image_input = process_image_for_model(image)
        image_input = list(img.to(device) for img in image_input)
        outputs = model.model(image_input)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        outputs_list.append(outputs)      

    pred_class, pred_mask, img_idx = process_outputs(outputs_list, images_list, object_class, instruction)
    # if mask for the object id is predicted, and mask is not empty
    if pred_class is not None:
        mask = pred_mask
        compressed_mask = compress_mask(mask)
        if compressed_mask:
            command[command["type"].lower()]["object"]["mask"] = compressed_mask
            command[command["type"].lower()]["object"]["colorImageIndex"] = img_idx
    return command


# predict action for the current utterance using the NS model
def predict_action_and_object_ns(model, input_dict, vocab, args, prev_action, obj_classes):
    with torch.no_grad():
        m_out = model.step(input_dict, vocab, prev_action=prev_action)

    m_pred = model_util.extract_action_preds(m_out, model.pad, vocab['action_low'], clean_special_tokens=False)[0]
    action_pred = m_pred['action']
    obj_pred = vocab['action_low'].index2word[m_pred['object'][0][0]]
    print("predicted action: %s, object: %s" % (action_pred, obj_pred))

    out = []
    # handle look around
    if action_pred == "look":
        action_item_dict = copy.deepcopy(LOOK_AROUND_ACTION)
        action_item_dict["id"] = str(uuid.uuid1())
        out.append(action_item_dict)

    # handle stop action
    elif action_pred == "<stop>":
        pass

    # # handle rotation
    elif action_pred == "rotate":    
        rotate_deg = "90"
        action_item_dict = copy.deepcopy(ROTATE_ACTION)
        action_item_dict["id"] = str(uuid.uuid1())
        action_item_dict["rotation"]["direction"] = "Right"
        action_item_dict["rotation"]["magnitude"] = rotate_deg
        out.append(action_item_dict)     

    # handle goto room action
    elif action_pred == "goto" and obj_pred in valid_rooms:
        action_item_dict = {
            "id": str(uuid.uuid1()),
            "type": "Goto", 
            "goto": {
                "object": {
                "officeRoom": obj_pred, 
                }
            }
        }
        out.append(action_item_dict)

    # handle other actions
    else:
        # recover the original object name from the merged one
        obj_pred_split = " ".join(obj_pred.split("_"))
        if obj_pred_split in obj_classes and not obj_pred_split == "Unassigned":
            action_item_dict = {
                "id": str(uuid.uuid1()),
                "type": action_pred.capitalize(), 
                action_pred: {
                    "object": {
                    "colorImageIndex": 0,
                    "name": obj_pred_split, 
                    }
                }
            }
            out.append(action_item_dict)

        else:
            # if not a valid prediction, skip to the next utterance
            print("Predicted action/object not valid, action %s, object %s" % (action_pred, obj_pred_split))     
    
    return out, action_pred, obj_pred