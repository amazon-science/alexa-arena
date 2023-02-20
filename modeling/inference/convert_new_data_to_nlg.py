# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import numpy as np
import json
import modeling.inference.constants as constants

'''
This script is used to convert the raw data to a data format ingestible 
by the models to run the mission level evaluation.

'''
with open(constants.validation_data) as f:
    data = json.load(f)

all_data = []
all_actions = set([])
all_objects = set([])

nlg_commands = []
ind_cnt = 1

for task_descr, task in data.items():
    human_annotations = task["human_annotations"]
    for annot_index, annotation in enumerate(human_annotations):
        di_item = {"test_number": ind_cnt, "mission_id": task_descr, "mission_cdf":  task["CDF"], "utterances": [], "action_index": [], "question_answers": []}
        ind_cnt += 1
        instructions = annotation["instructions"]
        for instruction in instructions:
            text_inst = instruction["instruction"].lower()
            question_answers = instruction.get("question_answers")
            if '_' in text_inst: #To get rid of occurences which have object IDs instead of objects
                continue
            question_answers = instruction.get("question_answers")
            if question_answers:
                di_item["question_answers"].append(question_answers)
            else:
                di_item["question_answers"].append([])
            di_item["utterances"].append(text_inst)
            di_item["annot_index"] = annot_index + 1
        nlg_commands.append(di_item)

np.save(constants.validation_numpy, nlg_commands)
