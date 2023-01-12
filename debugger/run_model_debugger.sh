# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


export PLATFORM="Linux"
export ML_TOOLBOX_DIR="$HOME/AlexaArena"
export PYTHONPATH="${PYTHONPATH}:${ML_TOOLBOX_DIR}"
echo "====== Starting model debugger ======"
python3 model_debugger.py
