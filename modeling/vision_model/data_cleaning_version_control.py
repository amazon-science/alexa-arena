# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import copy
import numpy as np
from PIL import Image
import os
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import json
import cv2
import base64

collapse_dicts = {
    "1" : {
        "Radio": ["Radio", "Broken Radio"],
        "Laser Tip": ["Laser Tip", "Broken Laser Tip"],
        "Floppy Disk": ["Floppy Disk", "Broken Virus Floppy Disk", "Broken AntiVirus Floppy Disk"],
        "Button": ["Button", "Buttons", "Green Button", "M8 Button", "Red Button", "Key Pad", "Blue Button", "B4 Button", "E7 Button", "E5 Button"],
        "Books": ["Book", "Magazine", "Books"],
        "Door": ["Door", "Garage Door", "Glass Doors"],
        "Printer Cartridge": ["Printer Cartridge", "Mug Printer Cartridge", "Figure Print Cartridge", "Hammer Print Cartridge", "Lever Print Cartridge"],
        "Sandwich": ["Sandwich", "Peanut Butter and Jelly Sandwich"],
        "Banana": ["Banana", "Banana Bunch"]
    },
    "2" : {
        "Table": ["Table", "Table Round Small"],
        "Apple": ["Apple", "Apple Slice"],
        "Sandwich": ["Burger", "Sandwich"],
        "Paper": ["Papers", "Stack of Papers", "Paper"],
        "Boxes": ["Boxes", "Pallet", "Cardboard Boxes",  "Box"],
        "Sign": ["Sign", "Wet Floor Sign", "Warning Sign"],
        "Cable": ["Cord", "Cable", "Cable Spool"],
        "Shelf": ["Shelf", "Red wall shelf", "Blue wall shelf", "Bookshelf"],
        "Cake": ["Cake", "Cake Slice"],
        "Sticky Note": ["Sticky Note", "Sticky Notes"],
        "Bread Slice": ["Bread Slice", "Loaf of Bread"],
        "Folder": ["Folder", "Folders"],
        "Pie": ["Pie", "Pie Slice"],
        "Mug": ["Mug", "Cup"],
        "Cabinet": ["Cabinet", "Server Cabinet"],
        "Handsaw": ["Handsaw", "Saw"],
        "Board": ["Cork Board", "Board"]
    },
    "3": {
        "Cooler": ["Cooler", "Water Cooler"]
    },
    "4":{
        "Tray": ["Tray", "Paper Tray"],
        "Computer": ["Computer", "Server"],
        "Counter": ["Counter", "Counter Top"]
    }
}

collapse_dict_v1 = collapse_dicts["1"]

collapse_dict_v2 = copy.deepcopy(collapse_dict_v1)
collapse_dict_v2.update(collapse_dicts["2"])
collapse_dict_v2["Sandwich"].append("Peanut Butter and Jelly Sandwich")


collapse_dict_v3 = copy.deepcopy(collapse_dict_v2)
collapse_dict_v3.update(collapse_dicts["3"])
collapse_dict_v3["Board"].extend(["White Board", "Whiteboard"])

collapse_dict_v4 = copy.deepcopy(collapse_dict_v3)
collapse_dict_v4.update(collapse_dicts["4"])
collapse_dict_v4["Sign"].append("Certificate")

collapse_dict_versions = {
    "v1": collapse_dict_v1,
    "v2": collapse_dict_v2,
    "v3": collapse_dict_v3,
    "v4": collapse_dict_v4,
}


def data_is_valid_v1(annotations_list, meta_path=None):
    new_bboxes = []
    for ann in annotations_list:
        bbox = ann["bbox"]
        assignment = ann["object_id"]
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        # print(area)
        rgb = ann["rgb"]
        if area > 0 and rgb != [0, 0, 0] and assignment != "Unassigned":
            new_bboxes.append(bbox)
    return bool(new_bboxes)

def read_image(path):
    im_frame = Image.open(path)
    np_frame = np.array(im_frame)[:,:,0:3]  # Leaving out the transperancy channel
    return np_frame

area_threshold_v2 = 80.0
def data_is_valid_v2(annotations_list):
    new_bboxes = []
    for ann in annotations_list:
        bbox = ann["bbox"]
        assignment = ann["object_id"]
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        rgb = ann["rgb"]
        if area > area_threshold_v2 and rgb != [0, 0, 0] and assignment != "Unassigned":
            new_bboxes.append(bbox)

    return bool(new_bboxes)

area_threshold_v3 = 120.0
# only checks if any valid annotations are present. on an image with anns below and above threshold it doesnt 
# remove the below threhsold objects. so additional data cleaning is necessary from the model data generator if
# we want to remove small objects from an image which has objects both above and below threshold.
# To reiterate, this only helps in removing images with all objcts below the threshold area. 
def data_is_valid_v3(annotations_list):
    new_bboxes = []
    for ann in annotations_list:
        bbox = ann["bbox"]
        assignment = ann["object_id"]
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        rgb = ann["rgb"]
        if area > area_threshold_v3 and rgb != [0, 0, 0] and assignment != "Unassigned":
            if bbox:
                new_bboxes.append(bbox)

    return bool(new_bboxes)


# Class based area pruning thresholds based computed thresholds
def data_is_valid_v4(annotations_list, object_id_to_class, class_to_area_thresholds):    
    new_bboxes = []
    for ann in annotations_list:
        bbox = ann["bbox"]
        object_id_long = ann["object_id"]
        object_id = '_'.join(object_id_long.split("_")[:-1])
        if object_id in object_id_to_class:
            class_assignment = object_id_to_class[object_id]
        else:
            if not object_id:
                if object_id_long == "Unassigned":
                    class_assignment = object_id_long
                else:
                    if "Chair" in object_id_long:
                        class_assignment = "Chair"
                    else:
                        class_assignment = object_id_to_class[object_id_long]
            else:
                # There are 4 so far: 
                if "Counter" in object_id:
                    class_assignment = "Counter"
                elif "Chair" in object_id:
                    class_assignment = "Chair"
                elif "sign_tall" in object_id or "sign_short" in object_id:
                    class_assignment = "Sign"
                elif "Lab_Terminal" in object_id:
                    class_assignment = "Machine Panel"
                else:
                    continue
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        rgb = ann["rgb"]
        try:
            area_threshold = class_to_area_thresholds[class_assignment][0]
        except:
            area_threshold = 1.0
        if area >= area_threshold and rgb != [0, 0, 0] and class_assignment != "Unassigned":
            if bbox:
                new_bboxes.append(bbox)

    return bool(new_bboxes)


# Class based area pruning thresholds based computed thresholds
def data_is_valid_v5(annotations_list, object_id_to_class, class_to_area_thresholds, trajectory_data_class_list): 
    new_bboxes = []
    for ann in annotations_list:
        bbox = ann["bbox"]
        object_id_long = ann["object_id"]
        object_id = '_'.join(object_id_long.split("_")[:-1])
        # object_id = object_id_long
        if object_id in object_id_to_class:
            class_assignment = object_id_to_class[object_id]
        else:
            class_assignment = \
                decode_absent_class_assignments(
                    object_id, object_id_long, object_id_to_class)
            if class_assignment == None:
                continue
        if class_assignment not in trajectory_data_class_list:
            class_assignment = "Unassigned"
        area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
        rgb = ann["rgb"]
        try:
            area_threshold = class_to_area_thresholds[class_assignment][0]
            area_threshold = min(100, area_threshold)
        except:
            area_threshold = 1

        if area >= area_threshold and rgb != [0, 0, 0] and class_assignment != "Unassigned" and class_assignment in trajectory_data_class_list:
            if bbox:
                new_bboxes.append(bbox)

    return bool(new_bboxes)

def decode_absent_class_assignments(object_id, object_id_long, object_id_to_class):
    class_assignment = None
    if not object_id:
        if object_id_long == "Unassigned":
            class_assignment = object_id_long
        else:
            if "Chair" in object_id_long:
                class_assignment = "Chair"
            else:
                if object_id_long in object_id_to_class:
                    class_assignment = object_id_to_class[object_id_long]
                else:
                    class_assignment = "Unassigned"
    else:
        # There are 5 so far:
        if "Counter" in object_id:
            class_assignment = "Counter"
        elif "Chair" in object_id:
            class_assignment = "Chair"
        elif "sign_tall" in object_id or "sign_short" in object_id:
            class_assignment = "Sign"
        elif "Lab_Terminal" in object_id:
            class_assignment = "Machine Panel"
        elif "ActionFigure" in object_id:
            class_assignment = "Action Figure"
        else:
            pass
    return class_assignment
