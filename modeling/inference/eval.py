# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1

import numpy as np
import torch
import argparse

from modeling.inference import constants

def get_test_data():
    TEST_NLG_COMMANDS = np.load(constants.validation_numpy, allow_pickle=True)
    return TEST_NLG_COMMANDS

def eval_ns():
    from modeling.inference.model_executors.ns_model_executor import ArenaNSModel
    torch.multiprocessing.set_start_method('spawn')
    arena_ns_model = ArenaNSModel(object_output_type="OBJECT_MASK") 
    arena_ns_model.evaluate_model(get_test_data())

def eval_vl():
    from modeling.inference.model_executors.vl_model_executor import ArenaVLModel
    # Use OBJECT_CLASS to test out language only model and switch to OBJECT_MASK to test the vision+language model
    arena_vl_model = ArenaVLModel(object_output_type="OBJECT_MASK")
    arena_vl_model.evaluate_model(get_test_data())

def eval_ph():
    from modeling.inference.model_executors.placeholder_model_executor import ArenaPHModel
    arena_ph_model = ArenaPHModel(object_output_type="OBJECT_MASK")
    arena_ph_model.evaluate_model(get_test_data())

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # Use "ns_model" to evaluate neural-symbolic model, "vl_model" for vision language model and "ph_model" for placeholder model
    parser.add_argument("--model", dest="model", type=str, default="ns_model", help="specify the model to evaluate")
    args = parser.parse_args()

    if args.model == "ns_model":
        eval_ns()
    elif args.model == "vl_model":
        eval_vl()
    elif args.model == "ph_model":
        eval_ph()
