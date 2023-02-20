# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


from arena_nn_model import ArenaNNModel
import os


TEST_NLG_COMMANDS = [
	{"test_number": 1, "mission_cdf_file_path": "cdf/cdf1.json", "utterances": ["Turn left"]},
	{"test_number": 2, "mission_cdf_file_path": "cdf/cdf1.json", "utterances": ["Turn right", "Turn right", "Turn right"
		, "Turn right", "Turn right", "Turn right"]},
	{"test_number": 3, "mission_cdf_file_path": "cdf/cdf1.json", "utterances": ["Turn left", "Turn left", "Turn left"
		, "Turn left", "Turn left", "Turn left", "Turn left", "Turn left"]}
]


def get_test_data():
	return TEST_NLG_COMMANDS


def main():
	data_path = os.getenv('ALEXA_ARENA_DIR') + "/data/demo-data"
	arena_nn_model = ArenaNNModel(object_output_type="OBJECT_CLASS", data_path=data_path)
	arena_nn_model.evaluate_model(get_test_data())
	print("Arena dependencies installation test is completed successfully")


if __name__ == '__main__':
	main()
