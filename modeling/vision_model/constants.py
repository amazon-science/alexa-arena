# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import os

REPO_ROOT = os.environ["HOME"]
DATA_ROOT = f"{REPO_ROOT}/AlexaArena/data/image_data/"
RG_OBJECT_LIST_ROOT = f"{REPO_ROOT}/AlexaArena/data/ObjectManifest/ObjectManifest.json"
CLASS_TO_AREA_THRESHOLDS_PATH = f"{REPO_ROOT}/AlexaArena/data/vision-data/class_to_area_thresholds_customized.json"
CLASSES_PATH = f"{REPO_ROOT}/AlexaArena/data/vision-data/classes.txt"
CUSTOM_CLASS_TO_OBJECT_ID_PATH = f"{REPO_ROOT}/AlexaArena/data/vision-data/class_to_obj_id_customized.json"
TRAINING_LOGS_ROOT = f"{REPO_ROOT}/training_logs/"
