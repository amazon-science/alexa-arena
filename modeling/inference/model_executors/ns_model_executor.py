# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import os
import sys
import copy
import time
import json
from PIL import Image
import torch
from dotmap import DotMap
import pickle

from arena_wrapper.arena_orchestrator import ArenaOrchestrator
from modeling.inference.models.ns_model import process_gt_mask, predict_action_and_object_ns
from modeling.inference.util import utils
from modeling.vision_model.models import MaskRCNN
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser
from modeling.ns_model.utils import data_util, model_util
from modeling.ns_model.nn.enc_visual import FeatureExtractor
from modeling.vision_model.constants import CUSTOM_CLASS_TO_OBJECT_ID_PATH
from modeling.ns_model.ns_data_generator import ArenaNSDataset
from modeling.ns_model.utils.utils import normalizeString

ALEXA_ARENA_DIR = os.getenv('ALEXA_ARENA_DIR')
CV_MODEL_CHECKPOINT_REL_PATH = os.path.join(ALEXA_ARENA_DIR, 'logs/vision_model_checkpt/21.pth')
CV_MODEL_NUM_CLASSES = 86
NS_MODEL_PATH = os.path.join(ALEXA_ARENA_DIR, "logs/ns_model_checkpt/1/")


class ArenaNSModel:
    def __init__(self, object_output_type):
        self.arena_orchestrator = ArenaOrchestrator()
        self.object_output_type = object_output_type
        self.process_mask = True  # Will be true if you have OBJECT_MASK flag set in eval.py and can set to False if using OBJECT_CLASS decoding
        self.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        self.cpu_device = torch.device("cpu")
        with open(NS_MODEL_PATH+"/config.json", "r") as f:
            args_saved = json.load(f)
        
        args = DotMap(args_saved)
        args.debug = False
        args.data_dir = os.getenv('ALEXA_ARENA_DIR')+ "/data/"
        args.checkpt_dir = os.getenv('ALEXA_ARENA_DIR') + "/logs/ns_model_checkpt/"
        args.vision_model_path = os.getenv('ALEXA_ARENA_DIR') + '/logs/vision_model_checkpt/21.pth'
        args.model_dir = os.getenv('ALEXA_ARENA_DIR')+"/modeling/"

        self.args = args
        self.max_steps = 50
        self.max_failed_actions = 10

        # load the pretrained model checkpt
        learned_model, _ = model_util.load_model(NS_MODEL_PATH+"/latest.pth", self.args.device)
        self.model = learned_model.model
        self.model.eval()
        self.model.args.device = self.args.device

        # load the training vocab
        with open(NS_MODEL_PATH+"%s_%s_vocabin.pkl" % (str(args.ann_type), "train"), "rb") as f:
            vocab_in = pickle.load(f)
        self.vocab = {'word': vocab_in, 'action_low': self.model.vocab_out}

        # use fasterrcnn pretrained on coco or maskrcnn pretrained on the vision dataset
        self.visual_archi = self.args.visual_archi
        if self.visual_archi == "fasterrcnn":
            self.extractor = FeatureExtractor(self.visual_archi, device=self.device, checkpoint=None, share_memory=True)
        elif self.visual_archi == "maskrcnn_v2":
            self.extractor = FeatureExtractor(self.visual_archi, device=self.device, \
                checkpoint=CV_MODEL_CHECKPOINT_REL_PATH, share_memory=True)

        with open(CUSTOM_CLASS_TO_OBJECT_ID_PATH, "r") as f:
            self.class2id = json.load(f)

    def load_cv_model(self, cv_model_path, num_classes):
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        args_vis = parser.parse_args()
        args_vis.visual_model = "maskrcnn"
        args_vis.gpu = True
        args_vis.dataset = "arena"
        args_vis.hidden_layer_size = 512
        args_vis.use_coco_pretrained_model = True
        vis_model = MaskRCNN(args=args_vis, num_classes=num_classes, device=self.device)
        vis_model.model.to(self.device)
        vis_model, _ = utils.restore_checkpoint(cv_model_path, vis_model, self.device)
        
        return vis_model

    def evaluate_model(self, test_data):
        # Start the unity instance
        if not self.arena_orchestrator.init_unity_instance():
            print("Could not start the unity process")
            return False
        cnt, failed_cnt, success_cnt = 0, 0, 0
        sum_sgcr = 0.0
        sum_steps = 0.0

        # Load CV model
        cv_model = self.load_cv_model(CV_MODEL_CHECKPOINT_REL_PATH, CV_MODEL_NUM_CLASSES)

        # Run tests
        mission_results = {}
        all_num_steps = {}
        for test_data_point in test_data:
            cnt += 1
            mission_id = test_data_point["mission_id"]
            if mission_id not in mission_results:
                mission_results[mission_id] = []

            if mission_id not in all_num_steps:
                all_num_steps[mission_id] = []

            goal_completion_status, subgoal_completion_rate, num_steps = self.run_test(test_data_point, cv_model)
            mission_results[mission_id].append(goal_completion_status)
            all_num_steps[mission_id].append(num_steps)

            sum_sgcr += subgoal_completion_rate
            sum_steps += num_steps
            print("Current mission subgoal completion rate: " + str(sum_sgcr / cnt))
            print("Current average number steps: " + str(sum_steps / cnt))
            if not goal_completion_status:
                failed_cnt += 1
                print("Num of failures: {}, Total run completed for: {}".format(failed_cnt, cnt))
            else:
                success_cnt += 1
                print("Num of successes: {}, Total run completed for: {}".format(success_cnt, cnt))
            
            print("-------------------------------------------------------")
            print("\n")
                

        print("Overall success rate: " + str((cnt - failed_cnt) / cnt))
        print("Overall subgoal completion rate: " + str(sum_sgcr / cnt))
        print("Overall average number of steps: " + str(sum_steps / cnt))

        with open(self.args.checkpt_dir + str(self.args.exp_num) + "/eval_succ.json", "w") as f:
            json.dump(mission_results, f, indent=4)

        with open(self.args.checkpt_dir + str(self.args.exp_num) + "/eval_steps.json", "w") as f:
            json.dump(all_num_steps, f, indent=4)    

        # Kill the unity instance
        if not self.arena_orchestrator.kill_unity_instance():
            print("Could not kill the unity instance. You might need to kill it manually")
        return True

    def run_test(self, test_data_point, cv_model):
        # Read CDF
        cdf = test_data_point["mission_cdf"]
        # Start the unity instance and launch a game
        if not self.arena_orchestrator.launch_game(cdf):
            print("Could not launch the game")
            return False
        time.sleep(10)
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
        
        goal_completion_status, subgoal_completion_status, num_steps = self.predict_and_execute_raw(test_data_point, cv_model)

        print("subgoal_completion_status: ", subgoal_completion_status)
        if goal_completion_status:
            print("Mission is completed for task: %s" % cdf["task_description"])
        else:
            print("Mission is not completed for task: %s" % cdf["task_description"])
        
        subgoal_completion_rate = sum(subgoal_completion_status) / float(len(subgoal_completion_status))
        return goal_completion_status, subgoal_completion_rate, num_steps

    def predict_and_execute_raw(self, test_data_point, cv_model):
        self.model.reset()
        failed_actions = 0
        t = 0
        prev_action = None
        look_around_imgs = []
        d_t = {}
        
        # fetch all the instructions and qas of an episode 
        all_lang = ""
        for u_idx, utt in enumerate(test_data_point["utterances"]):
            all_lang += " <<instr>> " + utt
            qas = test_data_point["question_answers"][u_idx]
            for qa in qas:
                qst = qa["question"].lower()
                qtype, q_obj = ArenaNSDataset.prep_question_tokens(qst)
                ans = qa["answer"]
                all_lang += qtype + q_obj + " <<ans>> " + ans

        d_t["lang"] = [normalizeString(all_lang)]
        print("Utterance: " +  d_t["lang"][0])
        
        while t < self.max_steps and failed_actions < self.max_failed_actions:
            print("current timestep: %s" % str(t))
            # since look around image is not the same as rotate image, apply some tricks to make sure we use look around images
            if prev_action == "look":
                look_around_imgs = [Image.fromarray(self.arena_orchestrator.get_images_from_metadata("colorImage")[c]) for c in range(4)]
                color_image = [look_around_imgs.pop(0)]
            elif prev_action == "rotate" and len(look_around_imgs) > 0:
                color_image = [look_around_imgs.pop(0)]
            else:
                color_image = [Image.fromarray(self.arena_orchestrator.get_images_from_metadata("colorImage")[0])]
                look_around_imgs = []
            
            # prepare vision data 
            d_t["frames"] = [data_util.extract_features(color_image, self.extractor)]
            input_dict, _ = data_util.tensorize_and_pad(d_t, self.vocab["word"], self.model.vocab_out, self.args.device, self.model.pad)
            
            command, action_pred, obj_pred = predict_action_and_object_ns(self.model, \
                input_dict, self.vocab, self.args, prev_action, list(self.class2id.keys()))

            if action_pred == "<stop>":
                break
            elif len(command) > 0:
                command = command[0]
                command = process_gt_mask(command, color_image, self.device, self.cpu_device, cv_model, d_t["lang"])

                return_val, error_code = self.arena_orchestrator.execute_action([command], self.object_output_type, d_t["lang"])
                if not return_val:
                    print("Action execution failed for utterance: ", d_t["lang"][0])
                    failed_actions += 1

            prev_action = str(action_pred)
            t += 1

        # return goal status
        _, goal_completion_status, subgoal_completion_status = self.arena_orchestrator.get_goals_status()
        return goal_completion_status, subgoal_completion_status, t
