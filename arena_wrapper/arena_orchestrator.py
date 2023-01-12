# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


from arena_wrapper.arena_controller import ArenaController
from arena_wrapper.arena_request_builder import ArenaRequestBuilder
from arena_wrapper.constants import ACTIONS_REQUIRING_MASK, OBJECT_CLASS_ALLOW_LIST
from arena_wrapper.enums.object_output_wrapper import ObjectOutputType
from arena_wrapper.util import object_class_decoder
import os
import subprocess
import atexit
import json
import time
import numpy as np
import base64
import cv2
import logging


class AppConfig:
    unity_executable_path = os.getenv('ARENA_PATH')
    unity_log_file = os.getenv('UNITY_LOG_PATH')
    host_pipe_file = os.getenv('HOST_PIPE')
    runtime_platform = os.getenv("PLATFORM")
    debug = True


class ArenaOrchestrator:
    def __init__(self, x_display=1):
        self.arena_request_builder = ArenaRequestBuilder()
        self.controller = ArenaController()
        self.x_display = x_display
        self.is_unity_running = False
        self.segmentation_images = None
        self.response = None
        self.segmentation_color_to_object_id_map = {}
        self.subgoals_completion_indices = []
        self.subgoals_completion_ids = []
        self.logger = logging.getLogger("ArenaOrchestrator")

    def init_game(self, cdf):
        if self.init_unity_instance():
            self.launch_game(cdf)
            return True
        return False

    def execute_action(self, actions, object_output_type, nlg_action):
        rg_compatible_actions = []
        try:
            if object_output_type == ObjectOutputType.OBJECT_CLASS:
                if not self.validate_object_classes(actions):
                    self.logger.error("Invalid object classes found. Not executing any actions.")
                    return False, "InvalidObjectClass"
                self.logger.debug("Cross-check against object class allow-list successful. Valid classes found.")
                actions = object_class_decoder.convert_object_class_to_id(actions, self.response, nlg_action)
                self.logger.info("Converted actions after decoding object classes: %s" % actions)
            params = {"segmentationImages": self.segmentation_images,
                      "segmentationColorToObjectIdMap": self.segmentation_color_to_object_id_map,
                      "objectOutputType": object_output_type,
                      "rightHandHeldObject": self.get_right_hand_held_object()}
            for action in actions:
                rg_compatible_action = self.arena_request_builder.get_request_json(action, params)
                print("RG compatible action: " + str(rg_compatible_action))
                if rg_compatible_action is not None:
                    rg_compatible_actions.append(rg_compatible_action)
        except Exception as e:
            self.logger.error("Skipping actions execution as exception occurred while interpreting actions: %s" % e)
            return False, "IncorrectActionFormat"
        if len(rg_compatible_actions) != 0:
            try:
                self.response = json.loads(self.controller.interact(rg_compatible_actions))
                self.segmentation_images = self.get_images_from_metadata("instanceSegmentationImage")
                self.build_segmentation_color_to_object_id_map()
                return self.response["lastActionSuccess"] == "ActionSuccessful", self.response["lastActionSuccess"]
            except Exception as ex:
                self.logger.error("Exception while executing actions: %s" % ex)
                return False, "ActionExecutionError"
        return False, "UnknownError"

    def stop_game(self):
        self.kill_unity_instance()

    def init_unity_instance(self):
        try:
            if self.is_unity_running:
                self.kill_unity_instance()
                time.sleep(1)
        except Exception as e:
            print("Exception occurred while killing the RG unity instance.", e)
        print("Starting unity instance...")
        env = os.environ.copy()
        env["DISPLAY"] = ":" + str(self.x_display)
        try:
            self.unity_proc = proc = subprocess.Popen(self._get_unity_execution_command(), env=env, shell=True)
            (output, err) = self.unity_proc.communicate()
            print(output)
            print(err)
            self.unity_pid = proc.pid
            print("Unity instance process ID: " + str(self.unity_pid))
            atexit.register(lambda: self.unity_proc.poll() is None and self.unity_proc.kill())
        except Exception as e:
            print("Exception occurred while opening the RG unity instance. Please start it. ", e)
            return False
        time.sleep(1)
        if not self.controller.get_connection_status():
            self.controller.start()
        return True

    def _get_unity_execution_command(self):
        command = None
        if AppConfig.runtime_platform == "Linux":
            command = 'DISPLAY=:' + str(self.x_display) + ' ' + AppConfig.unity_executable_path + ' -logfile ' + AppConfig.unity_log_file + '&'
        elif AppConfig.runtime_platform == "Mac":
            command = 'open -n ' + AppConfig.unity_executable_path
        return command

    def launch_game(self, cdf):
        self.controller.handle_init(cdf)
        time.sleep(10)
        return True

    def kill_unity_instance(self):
        try:
            os.system("kill -9 $(ps -A | grep Arena | awk '{ print $1 }')")
            print("Unity process killed successfully")
            return True
        except Exception as e:
            print("Exception occurred while killing the RG unity instance.", e)
            return False

    def build_segmentation_color_to_object_id_map(self):
        objects = self.response['objects']
        self.segmentation_color_to_object_id_map = {}
        for obj in objects:
            color_dict = obj['instanceSegmentationColor']
            color_dict.pop('a', None)
            key = frozenset(color_dict.items())
            if key not in self.segmentation_color_to_object_id_map:
                self.segmentation_color_to_object_id_map[key] = obj['objectID']
            else:
                self.logger.error(obj['instanceSegmentationColor'], " This color already exists for object: ",
                                  self.segmentation_color_to_object_id_map[key])

    def get_scene_data(self):
        exclude_keys = ["colorImage", "depthImage", "normalsImage", "instanceSegmentationImage"]
        return {key: self.response[key] for key in self.response if key not in exclude_keys}

    def get_images_from_metadata(self, image_key):
        if image_key not in self.response:
            return None
        image_rgb_list = []
        for raw_image in self.response[image_key]:
            encoded_image = str.encode(raw_image)
            decoded_image = base64.decodebytes(encoded_image)
            image_buffer = np.asarray(bytearray(decoded_image), dtype="uint8")
            image_bgr = cv2.imdecode(image_buffer, cv2.IMREAD_COLOR)
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            image_rgb_list.append(image_rgb)
        return image_rgb_list

    def get_goals_status(self):
        subgoals_completion_ids_in_current_action = []
        all_goals_completed = False
        subgoal_completion_status = []
        if "challengeProgress" not in self.response:
            return subgoals_completion_ids_in_current_action, all_goals_completed, subgoal_completion_status
        goals = self.response["challengeProgress"]["ChallengeGoals"]
        try:
            all_goals_completed = True
            for task in goals:
                goal_id = task["goal_id"]
                if task["isFinished"]:
                    subgoal_completion_status.append(1)
                    if goal_id not in self.subgoals_completion_ids:
                        self.logger.info(f"Task {goal_id} has been completed: {str(task)}")
                        subgoals_completion_ids_in_current_action.append(goal_id)
                        self.subgoals_completion_ids.append(goal_id)
                else:
                    subgoal_completion_status.append(0)
                    all_goals_completed = False
        except Exception as e:
            self.logger.error(f"Unable to get goal status: {str(e)}")
        return subgoals_completion_ids_in_current_action, all_goals_completed, subgoal_completion_status

    def validate_object_classes(self, actions):
        for action in actions:
            try:
                if action["type"] in ACTIONS_REQUIRING_MASK:
                    object_class_name = action[action["type"].lower()]["object"]["name"]
                    if object_class_name not in OBJECT_CLASS_ALLOW_LIST:
                        return False
            except Exception as ex:
                self.logger.error("Exception while validating object classes: %s" % ex)
                raise ex
        return True

    def get_reconstructed_metadata(self):
        robot_info = list()
        for obj in self.response["objects"]:
            if "TAM_" in obj["objectID"]:
                robot_info.append({
                    "currentRoom": obj["currentRoom"],
                    "position": obj["position"],
                    "rotation": obj["rotation"]
                })
                break
        if not robot_info:
            self.logger.error("TAM location not found in the object list")
        if "colorImage" not in self.response:
            self.logger.error("Color images not found in the RG response")
        color_images = {}
        for color_image_index, color_image in enumerate(self.response["colorImage"]):
            color_images[str(color_image_index)] = self.response["colorImage"][color_image_index]
        depth_images = {}
        for depth_image_index, depth_image in enumerate(self.response["depthImage"]):
            depth_images[str(depth_image_index)] = self.response["depthImage"][depth_image_index]
        return {
            "colorImages": color_images,
            "depthImages": depth_images,
            "robotInfo": robot_info,
            "viewPoints": self.response["sceneMetadata"]["GoToPoints"]
        }

    def get_right_hand_held_object(self):
        if self.response is None or "agent" not in self.response or "rightHandHeldObject" not in self.response["agent"]:
            return None
        return self.response["agent"]["rightHandHeldObject"]
