# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


from modeling.inference.model_executors.vl_model_executor import ArenaNNModel
import numpy as np
from modeling.inference import constants


def get_test_data():
    TEST_NLG_COMMANDS = np.load(constants.validation_numpy, allow_pickle=True)
    return TEST_NLG_COMMANDS

def main():
    # Use OBJECT_CLASS to test out language only model and switch to OBJECT_MASK to test the vision+language model
    arena_nn_model = ArenaNNModel(object_output_type="OBJECT_MASK")
    arena_nn_model.evaluate_model(get_test_data())


if __name__ == '__main__':
    main()
