# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import json


def map_arena_coco_common():
    COCO_INSTANCE_CATEGORY_NAMES = [
        '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
        'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'street sign', 'stop sign',
        'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
        'elephant', 'bear', 'zebra', 'giraffe', 'hat', 'backpack', 'umbrella', 'shoe', 'eye glasses',
        'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
        'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
        'bottle', 'plate', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
        'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
        'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'mirror', 'dining table',
        'window', 'desk', 'toilet', 'door', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
        'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'blender', 'book',
        'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
    ]
    coco_class_to_idx = {cat: idx for idx, cat in enumerate(COCO_INSTANCE_CATEGORY_NAMES)}
    print("Number of coco classes: ", len(COCO_INSTANCE_CATEGORY_NAMES))
    arena_to_coco_mapping_common = {
        "Unassigned": "__background__",
        "Action Figure": "person",
        "Apple": "apple",
        "Banana": "banana",
        "Books": "book",
        "Bowl": "bowl",
        "Cake": "cake",
        "Carrot": "carrot",
        "Chair": "chair",
        "Clock": "clock",
        "Computer": "tv",
        "Control Panel": "parking meter",
        "Couch": "couch",
        "Desk": "desk",
        "Donut": "donut",
        "Door": "door",
        "Fire Extinguisher": "fire hydrant",
        "Fork": "fork",
        "Fridge": "refrigerator",
        "Golf Ball": "sports ball",
        "Keyboard": "keyboard",
        "Knife": "knife",
        "Light": "traffic light",
        "Microwave": "microwave",
        "Mug": "cup",
        "Plant": "potted plant",
        "Plate": "plate",
        "Sandwich": "sandwich",
        "Sign": "street sign",
        "Sink": "sink",
        "Spoon": "spoon",
        "Table": "dining table",
        "Toaster": "toaster",
    }
    arena_to_coco_class_to_idx = {
        cat: coco_class_to_idx[arena_to_coco_mapping_common[cat]] for cat in arena_to_coco_mapping_common
    }
    arena_to_coco_class_to_idx["Printer"] = 100  # For all other classes, randomly choosing printer
    print("Number of common classes", len(arena_to_coco_mapping_common))
    # import pdb; pdb.set_trace()
    return arena_to_coco_class_to_idx

if __name__ == "__main__":
    arena_to_coco_class_to_idx = map_arena_coco_common()
    print(arena_to_coco_class_to_idx)
    # {'Unassigned': 0, 'Action Figure': 1, 'Apple': 53, 'Banana': 52, 'Books': 84, 'Bowl': 51, 'Cake': 61, 'Carrot': 57, 'Chair': 62, 'Clock': 85, 'Computer': 72, 'Control Panel': 14, 'Couch': 63, 'Desk': 69, 'Donut': 60, 'Door': 71, 'Fire Extinguisher': 11, 'Fork': 48, 'Fridge': 82, 'Golf Ball': 37, 'Keyboard': 76, 'Knife': 49, 'Light': 10, 'Microwave': 78, 'Mug': 47, 'Plant': 64, 'Plate': 45, 'Sandwich': 54, 'Sign': 12, 'Sink': 81, 'Spoon': 50, 'Table': 67, 'Toaster': 80}