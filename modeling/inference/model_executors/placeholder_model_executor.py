# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import os
from arena_wrapper.arena_orchestrator import ArenaOrchestrator

from modeling.inference.models.placeholder_model import *
from modeling.inference.util import utils
from modeling.vision_model.models import MaskRCNN
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


ML_TOOLBOX_BASE_DIR_PATH = os.getenv('ML_TOOLBOX_DIR')
# Please modify the path below to load your model
CV_MODEL_CHECKPOINT_REL_PATH = ML_TOOLBOX_BASE_DIR_PATH + '/logs/vision_model_checkpt/21.pth'
CV_MODEL_NUM_CLASSES = 86


class ArenaNNModel:
    def __init__(self, object_output_type, data_path=None):
        self.arena_orchestrator = ArenaOrchestrator()
        self.object_output_type = object_output_type
        self.data_path = data_path
        self.process_mask = True  # Will be true if you have OBJECT_MASK flag set in eval.py and can set to False if using OBJECT_CLASS decoding
        self.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.cpu_device = torch.device("cpu")

    def train(self):
        pass

    def predict(self, instruction, color_images, instruction_history='', cv_model=None):
        out, instruction_history = predict_action_and_object(instruction, instruction_history)
        if self.process_mask:  # True for OBJECT_MASK decoding, False for OBJECT_CLASS
            out = process_gt_mask(out, color_images, self.device, self.cpu_device,
                                  cv_model)  # Send the list of images for processing
        return out

    def evaluate_model(self, test_data):
        # Start the unity instance
        if not self.arena_orchestrator.init_unity_instance():
            print("Could not start the unity process")
            return False
        # time.sleep(10)
        cnt, failed_cnt, success_cnt = 0, 0, 0
        sum_sgcr = 0.0
        # Load CV model
        cv_model = self.load_cv_model(CV_MODEL_CHECKPOINT_REL_PATH, CV_MODEL_NUM_CLASSES)
        # Run tests
        for test_data_point in test_data:
            cnt += 1
            goal_completion_status, subgoal_completion_rate = self.run_test(test_data_point, cv_model)
            sum_sgcr += subgoal_completion_rate
            print("Current mission subgoal completion rate: " + str(sum_sgcr / cnt))
            if not goal_completion_status:
                failed_cnt += 1
                print("Num of failures: {}, Total run completed for: {}".format(failed_cnt, cnt))
            else:
                success_cnt += 1
                print("Num of successes: {}, Total run completed for: {}".format(success_cnt, cnt))

        print("Overall success rate: " + str((cnt - failed_cnt) / cnt))  # Measure and print mission level success rate
        print("Overall subgoal completion rate: " + str(sum_sgcr / cnt))  # Measure and print subgoal completion rate
        # Kill the unity instance
        if not self.arena_orchestrator.kill_unity_instance():
            print("Could not kill the unity instance. You might need to kill it manually")
        return True

    def read_cdf(self, cdf_file_name):
        if self.data_path is None:
            print("Data path not set")
            return None
        f = open(self.data_path + "/" + cdf_file_name)
        data = json.load(f)
        return data

    def load_default_cv_model(self):
        return self.load_cv_model(CV_MODEL_CHECKPOINT_REL_PATH, CV_MODEL_NUM_CLASSES)

    def load_cv_model(self, cv_model_path, num_classes):
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        args = parser.parse_args()
        args.visual_model = "maskrcnn"
        args.gpu = True
        args.dataset = "arena"
        args.hidden_layer_size = 512
        args.use_coco_pretrained_model = True
        model = MaskRCNN(args=args, num_classes=num_classes, device=self.device)
        model.model.to(self.device)
        model, _ = utils.restore_checkpoint(cv_model_path, model, self.device)
        
        return model

    def run_test(self, test_data_point, cv_model):
        print(test_data_point)
        # Read CDF
        cdf = test_data_point["mission_cdf"]
        # Start the unity instance and launch a game
        if not self.arena_orchestrator.launch_game(cdf):
            print("Could not launch the game")
            return False
        # TODO: Remove this sleep time if not needed on EC2 instance
        # time.sleep(10)
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
            print("Utterance: " + utterance)
            json_commands = self.predict(utterance, color_images, instruction_history='', cv_model=cv_model)
            print("Output: " + str(json_commands))
            return_val, error_code = self.arena_orchestrator.execute_action(json_commands, self.object_output_type,
                                                                            utterance)
            color_images = self.arena_orchestrator.get_images_from_metadata("colorImage")
            if not return_val:
                print("Action could not be completed for utterance: ", utterance)
        # Check goal status and return test result
        subgoals_completion_ids_in_current_action, goal_completion_status, subgoal_completion_status = self.arena_orchestrator.get_goals_status()
        print("subgoal_completion_status: ", subgoal_completion_status)
        if goal_completion_status:
            print("Mission is completed")
        else:
            print("Mission is not completed")
        subgoal_completion_rate = sum(subgoal_completion_status) / float(len(subgoal_completion_status))
        return goal_completion_status, subgoal_completion_rate
