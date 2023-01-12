# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import numpy as np
import onnx
import onnxruntime


path_to_model = "/home/ubuntu/training_logs/exp-1/model_checkpoints/0.onnx"
model = onnx.load(path_to_model)
onnx.checker.check_model(model)
sess = onnxruntime.InferenceSession(path_to_model,  providers=['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider'])
input_name = sess.get_inputs()[0].name
input_shape = sess.get_inputs()[0].shape
device = "cuda:0"
output_names = [output.name for output in sess.get_outputs()]
im = np.random.randn(input_shape[0], input_shape[1], input_shape[2], input_shape[3]).astype('f')
res = sess.run(output_names, {input_name: im})
print(res)
