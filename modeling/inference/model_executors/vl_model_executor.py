# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import copy
import os
import random
import string
import time
import json
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser

from arena_wrapper.arena_orchestrator import ArenaOrchestrator
from modeling.inference.models.vl_model import *
from modeling.inference.util.robot_actions import *
from modeling.vl_model.models import build_predictor

ALEXA_ARENA_DIR = os.getenv('ALEXA_ARENA_DIR')

VL_MODEL_CHECKPOINT_PATH = os.path.join(ALEXA_ARENA_DIR, "logs/vl_model_checkpt/65.pth")
CLIP_LOAD_MODEL_PATH = os.path.join(
    ALEXA_ARENA_DIR, "modeling/vl_model/pretrained/RN50.pt")
EVAL_AI_SAVE_FOLDER = "eval_ai_folder"
if not os.path.exists(EVAL_AI_SAVE_FOLDER):
    os.makedirs(EVAL_AI_SAVE_FOLDER)


class ArenaVLModel:
    def __init__(self, object_output_type, data_path=None):
        self.arena_orchestrator = ArenaOrchestrator()
        self.object_output_type = object_output_type
        self.data_path = data_path
        self.device = torch.device(f'cuda:0') if torch.cuda.is_available() else torch.device('cpu')
        id2class_path = os.path.join(
            ALEXA_ARENA_DIR,
            "data/vision-data/obj_id_to_class_customized.json"
        )
        class2idx_path = os.path.join(
            ALEXA_ARENA_DIR,
            "data/vision-data/class_to_idx.json"
        )
        with open(class2idx_path) as f:
            self.class2idx = json.load(f)
        room_objs = {
            'BreakRoom': 86,
            'Reception': 87,
            'Lab2': 88,
            'Lab1': 89,
            'SmallOffice': 90,
            'MainOffice': 91,
            'Left': 92,
            'Right': 93
        }
        self.class2idx.update(room_objs)
        with open(id2class_path) as f:
            self.id2class = json.load(f)
        self.act2idx = {
            'Pickup': 0,
            'Break': 1,
            'Close': 2,
            'Open': 3,
            'Pour': 4,
            'Scan': 5,
            'Goto': 6,
            'Place': 7,
            'Toggle': 8,
            'Clean': 9,
            'Fill': 10,
            'Rotate': 11
        }
        self.idx2act = {v: k for k, v in self.act2idx.items()}
        self.idx2obj = {v: k for k, v in self.class2idx.items()}

    def evaluate_model(self, test_data):
        # Start the unity instance
        if not self.arena_orchestrator.init_unity_instance():
            print("Could not start the unity process")
            return False
        cnt, failed_cnt, success_cnt = 0, 0, 0
        sum_sgcr = 0.0
        # Load VL model
        vl_model = self.load_vl_model(VL_MODEL_CHECKPOINT_PATH)
        # Run test
        for test_data_point in test_data:
            cnt += 1
            goal_completion_status, subgoal_completion_rate, steps = self.run_test(test_data_point, vl_model)
            sum_sgcr += subgoal_completion_rate
            print("Current mission subgoal completion rate: " + str(sum_sgcr / cnt))
            print("Number of steps to complete mission: {}".format(steps))
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
        # Kill the unity instance
        if not self.arena_orchestrator.kill_unity_instance():
            print("Could not kill the unity instance. You might need to kill it manually")
        return True

    def load_vl_model(self, vl_model_path):
        parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
        args = parser.parse_args()
        args.word_len = 22
        args.clip_pretrain = CLIP_LOAD_MODEL_PATH
        args.fpn_in = [512, 1024, 1024]
        args.fpn_out =  [256, 512, 1024]
        args.num_layers = 3
        args.vis_dim = 512
        args.word_dim = 1024
        args.num_head = 8
        args.dim_ffn = 2048
        args.dropout = 0.1
        args.lr_multi = 0.0
        args.base_lr = 0.0
        args.lr_seq = 0.0
        args.intermediate = False
        model, _ = build_predictor(args)
        checkpoint = torch.load(vl_model_path, map_location=torch.device(torch.device('cuda:0')))
        model.load_state_dict(checkpoint['model'], strict=True)
        model.eval()
        model.to(self.device)
        return model

    def run_test(self, test_data_point, cv_model):
        # Read CDF
        cdf = test_data_point["mission_cdf"]
        # Start the unity instance and launch a game
        if not self.arena_orchestrator.launch_game(cdf):
            print("Could not launch the game")
            return False
        time.sleep(5)
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
        goal_completion_status, subgoal_completion_status, steps = self.predict_and_execute_raw(test_data_point, cv_model)

        print("subgoal_completion_status: ", subgoal_completion_status)
        if goal_completion_status:
            print("Mission is completed for task: %s" % test_data_point["mission_cdf"]["task_description"])
        else:
            print("Mission is not completed for task: %s" % test_data_point["mission_cdf"]["task_description"])
        
        subgoal_completion_rate = sum(subgoal_completion_status) / float(len(subgoal_completion_status))
        return goal_completion_status, subgoal_completion_rate, steps
    
    def robot_stuck_in_loop(self, prediction_history, chunk_size=1):
        if len(prediction_history) < 10:
            return False
        near_history = prediction_history[-10:]
        if chunk_size == 1: # Last 10 same actions
            return all(x == near_history[0] for x in near_history)
        if chunk_size == 2: # Last 5 pairs of actions are the same in sequence
            odd_history = [elt for i, elt in enumerate(near_history) if ((i + 1) % 2 != 0)]
            even_history =  [elt for i, elt in enumerate(near_history) if ((i + 1) % 2 == 0)]
            odd_ = all(x == odd_history[0] for x in odd_history)
            even_ = all(x == even_history[0] for x in even_history)
            return (odd_ and even_)
        return False

    def get_question_type(self, qst):
        qst = qst.lower()
        if "where is" in qst:
            qtype = " <<loc>> "
            q_obj = qst.split()[-1][:-1]
        elif "look like" in qst:
            qtype = " <<app>> "
            q_obj = qst.split()[3]
        elif "referring to" in qst:
            qtype = " <<ref>> "
            q_obj = qst.split()[1]
        elif "which direction" in qst:
            qtype = " <<dir>> "
            q_obj = ""
        else:
            # include the whole sentence for free-form questions
            qtype = " <<q>> "
            q_obj = qst
        return (qtype, q_obj)

    def predict_and_execute_raw(self, test_data_point, cv_model):
        # Get color image from dummy action
        color_images = self.arena_orchestrator.get_images_from_metadata("colorImage")
        # Iterate over instructions and execute predicted actions on Unity instance
        utterances = test_data_point["utterances"]
        mission_id = test_data_point["mission_id"]
        question_answers = test_data_point.get("question_answers")
        annot_num = test_data_point["annot_index"]
        max_steps = 50
        max_num_failed_actions = 10
        steps, failed_steps = 0, 0
        prediction_history = []
        eval_ai_predictions_dict = {"predicted_actions": [], "last_game_state": None}
        # Iterating through every utterance in the mission one by one through the model
        for u_idx, utterance in enumerate(utterances):
            answers_list = []
            question_answers_list = question_answers[u_idx]
            if question_answers_list:
                answers_list = [qa['answer'].strip(string.punctuation).lower() for qa in question_answers_list]
                questions = [qa['question'] for qa in question_answers_list]
                qtype_objs = [self.get_question_type(qst) for qst in questions]
                random_index = random.sample([x for x in range(len(answers_list))], k=1)[0]
                answers = [answers_list[random_index]]
                answers_list = [qtype_objs[random_index][0].strip()] 
                answers_list.extend(answers)
                answers_string = ' '.join(answers_list)
            else:
                answers_string = ''
                
            print("Utterance: " + utterance)
            augmented_utterance = (utterance.strip(string.punctuation) + ' ' + answers_string).strip()
            print("Utterance + answers: ", augmented_utterance)
            print('\n')
            # Executing the the first predicted command
            command, eos, ht, ct = predict_action_and_object_e2e_vl(
                    vl_model=cv_model,
                    images_list=color_images,
                    instruction=augmented_utterance,
                    eos=True,
                    ht_1=None,
                    ct_1=None,
                    id2act= self.idx2act,
                    id2obj=self.idx2obj,
                    device=self.device)
            history_comm = copy.deepcopy(command)
            history_comm.pop('id')
            prediction_history.append(history_comm)
            chunk_1_loop = self.robot_stuck_in_loop(prediction_history, chunk_size=1)
            chunk_2_loop = self.robot_stuck_in_loop(prediction_history, chunk_size=2)
            if chunk_1_loop or chunk_2_loop:
                eos = False  # EOS is false but the hidden state in the lstm needs to be reset to start anew for the same instruction
                ht, ct = None, None
                prediction_history = []
                command = {"id": "", "type": "Rotate",  "rotation": {"direction": "Right", "magnitude": 10}}
            return_val, error_code = self.arena_orchestrator.execute_action([command], self.object_output_type, utterance)
            steps += 1
            eval_ai_predictions_dict['predicted_actions'].append(command)
            if error_code != "ActionSuccessful":
                failed_steps += 1
            if (steps > max_steps) or (failed_steps > max_num_failed_actions):
                break
            color_images = self.arena_orchestrator.get_images_from_metadata("colorImage")
            if not return_val:
                print("Action execution failed for utterance: ", utterance)

            # If last action was not eos
            while not eos:
                command, eos, ht, ct = predict_action_and_object_e2e_vl(
                    vl_model=cv_model,
                    images_list=color_images,
                    instruction=augmented_utterance,
                    eos=eos,
                    ht_1=ht,
                    ct_1=ct,
                    id2act= self.idx2act,
                    id2obj=self.idx2obj,
                    device=self.device)
                history_comm = copy.deepcopy(command)
                history_comm.pop('id')
                prediction_history.append(history_comm)
                chunk_1_loop = self.robot_stuck_in_loop(prediction_history, chunk_size=1)
                chunk_2_loop = self.robot_stuck_in_loop(prediction_history, chunk_size=2)
                if chunk_1_loop or chunk_2_loop:
                    # EOS is false but the hidden state in the lstm needs to be reset to start anew for the same instruction
                    eos = False
                    ht, ct = None, None
                    prediction_history = []
                    command = {"id": "", "type": "Rotate",  "rotation": {"direction": "Right", "magnitude": 10}}                
                return_val, error_code = self.arena_orchestrator.execute_action([command], self.object_output_type, utterance)
                steps += 1
                eval_ai_predictions_dict['predicted_actions'].append(command)
                if error_code != "ActionSuccessful":
                    failed_steps += 1
                if (steps > max_steps) or (failed_steps > max_num_failed_actions):
                    break
                color_images = self.arena_orchestrator.get_images_from_metadata("colorImage")
                if not return_val:
                    print("Action execution failed for utterance: ", utterance)
        eval_ai_predictions_dict["last_game_state"] = self.arena_orchestrator.response
        with open("{}/{}_{}.json".format(EVAL_AI_SAVE_FOLDER, mission_id, annot_num), 'w') as f:
            json.dump(eval_ai_predictions_dict, f, indent=2)
        _, goal_completion_status, subgoal_completion_status = self.arena_orchestrator.get_goals_status()
        return goal_completion_status, subgoal_completion_status, steps
