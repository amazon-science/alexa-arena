# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import json
import logging
import os
import cv2
import time

from arena_wrapper.arena_orchestrator import ArenaOrchestrator
from arena_wrapper.enums.object_output_wrapper import ObjectOutputType

# Constants used in defining log formatter
BLUE_COLOR_STRING = '\x1b[38;5;39m'
RESET_COLOR = '\x1b[0m'


class ArenaDebugger:
    def __init__(self):
        self.arena_orchestrator = ArenaOrchestrator()
        self.logger = logging.getLogger("ArenaDebugger")
        self.logger.setLevel(logging.DEBUG)
        log_formatter = logging.Formatter(fmt='%(message)s')
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(log_formatter)
        self.logger.addHandler(console_handler)
        self.metadata_dir_path = "/tmp/"

    def launch_game(self, cdf_file_path):
        try:
            # read CDF file
            with open(cdf_file_path) as f:
                cdf_data = json.load(f)
            self.metadata_dir_path = self.metadata_dir_path + cdf_data["scene"]["scene_id"]
            self.metadata_dir_path = self.metadata_dir_path.replace(" ", "")
            if not os.path.exists(self.metadata_dir_path):
                os.makedirs(self.metadata_dir_path)
            # init the game using orchestrator
            self.arena_orchestrator.init_game(cdf_data)
            # execute dummy action to get initial set of images
            return_val, error_code = self.execute_dummy_action()
            if not return_val:
                self.logger.error("Error in executing initial dummy action %s" % error_code)
            # Dump color image from dummy action
            color_images = self.arena_orchestrator.get_images_from_metadata("colorImage")
            self.save_images(color_images, "color_image_0")
            return True
        except Exception as ex:
            self.logger.error("Error in initializing the game: %s" % ex)
        return False

    def execute_action_sequence(self, actions_file_path, action_count):
        try:
            # read the list of actions
            with open(actions_file_path) as f:
                actions = json.load(f)
            # execute actions on arena
            ret_val, error_code = self.arena_orchestrator.execute_action(actions, ObjectOutputType.OBJECT_MASK, None)
            self.logger.info("Error code after executing action: %s" % (BLUE_COLOR_STRING + error_code + RESET_COLOR))
            if not ret_val:
                self.logger.error("Action could not be executed successfully.")
            # Dump color image
            color_images = self.arena_orchestrator.get_images_from_metadata("colorImage")
            self.save_images(color_images, "color_image_" + str(action_count))
        except Exception as ex:
            self.logger.error("Error in initializing the game: %s" % ex)
        return False

    def execute_dummy_action(self):
        dummy_action = [{
            "id": "1",
            "type": "Rotate",
            "rotation": {
                "direction": "Right",
                "magnitude": 0,
            }
        }]
        return_val, error_code = False, None
        action_counter = 0
        while not return_val and action_counter <= 10:
            return_val, error_code = self.arena_orchestrator.execute_action(dummy_action, ObjectOutputType.OBJECT_MASK, None)
            time.sleep(1)
            action_counter += 1
        return return_val, error_code

    def save_images(self, images, image_name_prefix):
        for image_index in range(len(images)):
            image = images[image_index]
            image_name = self.metadata_dir_path + "/" + image_name_prefix + "_" + str(image_index) + ".png"
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            cv2.imwrite(image_name, image_rgb)
            self.logger.info("Saving image at path: %s", BLUE_COLOR_STRING + image_name + RESET_COLOR)


def main():
    arena_debugger = ArenaDebugger()
    # Launch the game
    cdf_file_path = input("Enter the CDF file path: ")
    ret_val = arena_debugger.launch_game(cdf_file_path)
    if not ret_val:
        print("Error in launching the game. Please verify the CDF or dependencies installation")
        return
    # Start streaming on web browser
    input("Start streaming on web browser. Press 'Enter' to continue...")
    # Execute actions one by one
    action_count = 0
    while True:
        action_count += 1
        actions_file_path = input("Enter the actions file path (Press ctrl + c to exit): ")
        arena_debugger.execute_action_sequence(actions_file_path, action_count)


if __name__ == '__main__':
    main()
