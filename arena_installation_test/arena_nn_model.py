# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


from arena_wrapper.arena_orchestrator import ArenaOrchestrator
import json
import time


class ArenaNNModel:
	def __init__(self, object_output_type, data_path):
		self.arena_orchestrator = ArenaOrchestrator()
		self.object_output_type = object_output_type
		self.data_path = data_path

	def train(self):
		pass

	def predict(self, utterance, color_images):
		model = {}
		model["Turn left"] = {
			"id": "1",
			"type": "Rotate",
			"rotation": {
				"direction": "Left",
				"magnitude": 90,
			}
		}
		model["Turn right"] = {
			"id": "1",
			"type": "Rotate",
			"rotation": {
				"direction": "Right",
				"magnitude": 90,
			}
		}
		if utterance not in model:
			print("This model cannot predict actions for given input")
			return []
		return [model[utterance]]

	def evaluate_model(self, test_data):
		# Start the unity instance
		if not self.arena_orchestrator.init_unity_instance():
			print("Could not start the unity process")
			return False
		time.sleep(10)
		# Run tests
		for test_data_point in test_data:
			if not self.run_test(test_data_point):
				print("Test failed: ", test_data_point)
		# Kill the unity instance
		if not self.arena_orchestrator.kill_unity_instance():
			print("Could not kill the unity instance. You might need to kill it manually")
		return True

	def read_cdf(self, cdf_file_name):
		f = open(self.data_path + "/" + cdf_file_name)
		data = json.load(f)
		return data

	def run_test(self, test_data_point):
		print(test_data_point)
		# Read CDF
		cdf = self.read_cdf(test_data_point["mission_cdf_file_path"])
		# Start the unity instance and launch a game
		if not self.arena_orchestrator.launch_game(cdf):
			print("Could not launch the game")
			return False
		time.sleep(15)
		# Run a dummy action to get images and metadata
		dummy_action = [{
			"id": "1",
			"type": "Rotate",
			"rotation": {
				"direction": "Right",
				"magnitude": 0,
			}
		}]
		return_val = False
		for i in range(10):
			return_val, error_code = self.arena_orchestrator.execute_action(dummy_action, self.object_output_type, "Rotate right")
			if not return_val:
				print("Could not execute dummy action successfully at attempt: " + str(i))
				time.sleep(5)
			else:
				break
		if not return_val:
			print("Exhausted all retry attempts to execute the action")
			return False
		# Get color image from dummy action
		color_images = self.arena_orchestrator.get_images_from_metadata("colorImage")
		# Iterate over nlg instructions and execute them on Unity instance
		utterances = test_data_point["utterances"]
		for utterance in utterances:
			json_commands = self.predict(utterance, color_images)
			return_val, error_code = self.arena_orchestrator.execute_action(json_commands, self.object_output_type, utterance)
			if not return_val:
				print("Action could not be completed for utterance: ", utterance)
		# Check goal status and return test result
		subgoals_completion_ids_in_current_action, goal_completion_status, subgoal_completion_status = self.arena_orchestrator.get_goals_status()
		if not goal_completion_status:
			print("Mission is not completed")
			return False
		print("Mission is completed")
		return True
