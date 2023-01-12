# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import logging
import time
import json
import os

from arena_wrapper.arena_orchestrator import ArenaOrchestrator
from arena_wrapper.enums.object_output_wrapper import ObjectOutputType
from modeling.inference.model_executors.placeholder_model_executor import ArenaNNModel

BASE_DIR = os.environ['SUBMISSIONS_DIR_PATH']


class ArenaHelper:
    def __init__(self):
        self.arena_orchestrator = ArenaOrchestrator()
        self.model_handler = ArenaNNModel(object_output_type="OBJECT_MASK", data_path=None)
        self.cv_model = self.model_handler.load_default_cv_model()
        self.logger = logging.getLogger("ArenaHelper")
        self.logger.setLevel(logging.DEBUG)
        self.color_images = None
        self.predicted_actions_list = None

    def launch_game(self, cdf_data):
        try:
            self.predicted_actions_list = []
            self.arena_orchestrator.init_game(cdf_data)
            return_val, error_code = self.execute_dummy_action()
            if not return_val:
                self.logger.error("Error in executing initial dummy action %s" % error_code)
                return False
            else:
                self.logger.info("Game is initialized successfully")
            self.color_images = self.arena_orchestrator.get_images_from_metadata("colorImage")
            return True
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
            self.logger.info(f"Error code: {error_code}")
        return return_val, error_code

    def execute_instruction(self, utterance):
        try:
            actions = self.model_handler.predict(utterance, self.color_images, instruction_history='', cv_model=self.cv_model)
            self.predicted_actions_list.extend(actions)
            # execute actions on arena
            ret_val, error_code = self.arena_orchestrator.execute_action(actions, ObjectOutputType.OBJECT_MASK, None)
            if not ret_val:
                self.logger.error("Action could not be executed successfully.")
            self.color_images = self.arena_orchestrator.get_images_from_metadata("colorImage")
            return self.arena_orchestrator.get_scene_data()
        except Exception as ex:
            self.logger.error("Error in initializing the game: %s" % ex)
        return None

    def get_predicted_actions(self):
        return self.predicted_actions_list

    def get_latest_game_state(self):
        exclude_keys = ["sceneMetadata", "colorImage", "depthImage", "normalsImage", "instanceSegmentationImage", "objects"]
        return {key: self.arena_orchestrator.response[key] for key in self.arena_orchestrator.response if key not in exclude_keys}


def run_instructions(instructions, arena_helper):
    for data in instructions:
        command = data["instruction"]
        arena_helper.execute_instruction(command)
    response = {
        "predicted_actions": arena_helper.get_predicted_actions(),
        "last_game_state": arena_helper.get_latest_game_state()
    }
    return response


def generate_submission_file(test_case_responses, test_case_id, index):
    if not os.path.exists(BASE_DIR):
        os.makedirs(BASE_DIR)
        print("The new directory is created!")
    file_path = BASE_DIR + test_case_id + "_" + str(index) + ".json"
    with open(file_path, "w") as actions_file:
        actions_file.write(json.dumps(test_case_responses))


def main():
    arena_helper = ArenaHelper()
    with open('sample_test_data.json', 'r') as openfile:
        test_data = json.load(openfile)
    test_case_ids = test_data.keys()
    for test_case_id in test_case_ids:
        test_case = test_data[test_case_id]
        human_annotations = test_case["human_annotations"]
        cdf_data = test_case["CDF"]
        for index, human_annotation in enumerate(human_annotations):
            if arena_helper.launch_game(cdf_data):
                result = run_instructions(human_annotation["instructions"], arena_helper)
                generate_submission_file(result, test_case_id, index)
            else:
                print("Error while launching the game")


if __name__ == '__main__':
    main()
