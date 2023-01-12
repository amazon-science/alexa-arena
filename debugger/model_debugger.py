# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import cv2
import json
import os
import logging

from modeling.inference.model_executors.placeholder_model_executor import ArenaNNModel

# Constants used in defining log formatter
YELLOW_COLOR_STRING = '\x1b[38;5;226m'
GREEN_COLOR_STRING = '\x1b[32m'
RESET_COLOR = '\x1b[0m'


class ModelDebugger:
    def __init__(self):
        self.model_handler = ArenaNNModel(object_output_type="OBJECT_MASK", data_path=None)
        self.cv_model = self.model_handler.load_default_cv_model()
        self.logger = logging.getLogger("ModelDebugger")
        self.logger.setLevel(logging.DEBUG)
        log_formatter = logging.Formatter(fmt='%(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)

    def predict_actions(self, utterance, color_image_file_paths):
        color_images = []
        for file_path in color_image_file_paths:
            image = cv2.imread(file_path, cv2.IMREAD_COLOR)
            color_images.append(image)
        actions = self.model_handler.predict(utterance, color_images, instruction_history='', cv_model=self.cv_model)
        self.logger.info("Predicted actions:\n%s" % (GREEN_COLOR_STRING + json.dumps(actions, indent=2) + RESET_COLOR))
        return actions


def main():
    model_debugger = ModelDebugger()

    # Predict actions for input user utterance
    action_count = 0
    actions_file_base_dir = input("Enter a directory path for storing actions: ")
    actions_file_base_dir = actions_file_base_dir.replace(" ", "")
    if not os.path.exists(actions_file_base_dir):
        os.makedirs(actions_file_base_dir)
    while True:
        action_count += 1
        utterance = input("Please enter user utterance (Press ctrl + c to exit): ")
        # Pro-tip: 'color_image_file_paths' could be extended to input multiple images in order to
        # support 'LookAround' action.
        color_image_file_paths = [input("Enter the color image path: ")]
        actions = model_debugger.predict_actions(utterance, color_image_file_paths)
        actions_file_path = actions_file_base_dir + "/actions_" + str(action_count) + ".json"
        with open(actions_file_path, "w") as actions_file:
            actions_file.write(json.dumps(actions))
        print("Actions are stored in file %s" % (YELLOW_COLOR_STRING + actions_file_path + RESET_COLOR))


if __name__ == '__main__':
    main()
