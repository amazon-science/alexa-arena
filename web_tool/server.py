# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import logging
import json
import time

from gevent.pywsgi import WSGIServer
from flask import Flask, request
from flask_cors import CORS
import os
from os import listdir
from os.path import isfile, join

from arena_wrapper.arena_orchestrator import ArenaOrchestrator
from arena_wrapper.enums.object_output_wrapper import ObjectOutputType
from modeling.inference.model_executors.placeholder_model_executor import ArenaPHModel


class Controller:
    def __init__(self):
        self.model_handler = ArenaPHModel(object_output_type="OBJECT_MASK", data_path=None)
        self.cv_model = self.model_handler.load_default_cv_model()
        self.arena_orchestrator = ArenaOrchestrator()
        cdf_dir_path = os.environ['CDF_DIR_PATH']
        self.cdf_files = [join(cdf_dir_path, f) for f in listdir(cdf_dir_path) if (isfile(join(cdf_dir_path, f)) and ".json" in f)]
        cdf_file_path = self.cdf_files[0]
        with open(cdf_file_path) as f:
            input_cdf_data = json.load(f)
        self.arena_orchestrator.init_game(input_cdf_data)
        return_val, error_code = self.execute_dummy_action()
        if not return_val:
            print("Error in executing initial dummy action %s" % error_code)
        self.color_images = self.arena_orchestrator.get_images_from_metadata("colorImage")
        print("Server initialization complete")

    def begin_session(self):
        return {"uid": "Robot", "text": "Hey, What's up?"}

    def process_utterance(self):
        content = request.get_json()
        utterance = content["utterance"]
        actions = self.model_handler.predict(utterance, self.color_images, instruction_history='', cv_model=self.cv_model)
        return_val, error_code = self.arena_orchestrator.execute_action(actions, ObjectOutputType.OBJECT_MASK, None)
        if return_val:
            return {"uid": "Robot", "text": "Ok! What's next?"}
        return {"uid": "Robot", "text": "Sorry, I couldn't do it."}

    def get_cdfs(self):
        cdfs = []
        for cdf_file in self.cdf_files:
            with open(cdf_file) as f:
                cdf_data = json.load(f)
                cdfs.append({"id": cdf_data["scene"]["scene_id"], "text": cdf_data["scene"]["scene_id"], "cdf": cdf_data})
        return {"cdfs": cdfs}

    def start_game(self):
        content = request.get_json()
        cdf_data = content["cdf_data"]
        self.arena_orchestrator.launch_game(cdf_data)
        return_val, error_code = self.execute_dummy_action()
        if not return_val:
            print("Error in executing initial dummy action %s" % error_code)
        self.color_images = self.arena_orchestrator.get_images_from_metadata("colorImage")
        return "Ok"

    def ping(self):
        return "pong"

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


class App:
    def __init__(self):
        self._flask_app = Flask(__name__)
        CORS(self._flask_app)
        self.http_server = None
        self.controller = Controller()
        self.logger = logging.getLogger("App")

    def configure(self):
        self._flask_app.add_url_rule('/ping', 'ping', self.controller.ping, methods=["GET"])
        self._flask_app.add_url_rule('/begin_session', 'begin_session', self.controller.begin_session, methods=["POST"])
        self._flask_app.add_url_rule('/process_utterance', 'process_utterance', self.controller.process_utterance, methods=["POST"])
        self._flask_app.add_url_rule('/get_cdfs', 'get_cdfs', self.controller.get_cdfs, methods=["POST"])
        self._flask_app.add_url_rule('/start_game', 'start_game', self.controller.start_game, methods=["POST"])

    def get_flask_app(self):
        return self._flask_app

    def run(self):
        flask_port = 11000
        self.http_server = WSGIServer(("", flask_port), self._flask_app, log=self.logger, error_log=self.logger)
        self.http_server.serve_forever()


app = App()


def main():
    app.configure()
    app.run()


if __name__ == '__main__':
    main()
