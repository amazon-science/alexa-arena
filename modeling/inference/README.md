# Instance set up

This README explains how to run evaluation for a model at a mission level. This readme will go over the following steps:

1. Setup requirements

2. Take raw user utterance data and convert it to a data format accepted by the model

3. Run mission level evaluation to test if a given set of instructions will be able to complete a mission using a specific model


# Setting up environment

This environment setup assumes Ubuntu 18.04 environment.  It also assumes that the dependencies for the Arena executable were also installed prior to testing this out (covered in the top level README of this repository and tested in the `arena_installation_test` folder).


# Data Conversion
## 1. Download the required data

Use the script `scripts/fetch_trajectory_data.sh` to download the data. This populates the data in `~/AlexaArena/data/trajectory-data` and extracts it. Please change the paths in the fetch data script as per your setup.


## 2. Convert data to required format
The raw files (`train.json`, `valid.json`) can be converted to the output format by using the `convert_new_data_to_nlg.py` which ingests the raw files (e.g., `valid.json`) and outputs a numpy file (e.g. `nlg_commands_val.npy`). 
This script should be run from the `modeling/inference` folder and completed before the mission level evaluation is executed.
```
python convert_new_data_to_nlg.py
``` 

# Run mission level evaluation

Once the data is downloaded and converted, we can run the mission level evalaution which uses the utterances from the data and runs them through the model to convert them to executable actions which are then sent to the Arena environment to execute and evaluate whether the mission could be completed using those set of utterances and for the provided model. 

For running the evaluation with pretrained models, follow the steps for the respective models. The models are in `modeling/inference/models/` and the corresponding model executors are in `modeling/inference/model_executors/`.

## Placeholder model 
This uses rule based language parsing and uses the vision model for mask generation. 

1. Download the vision model using `scripts/fetch_vision_model.sh`
2. Double check the model checkpoint path in `modeling/inference/model_executors/placeholder_model_executor.py`
3. The model and corresponding executor are defined in `modeling/inference/models/placeholder_model.py` and `modeling/inference/model_executors/placeholder_model_executor.py` respectively.

## VL model 
This is an end-to-end vision language model. 
1. Download the trained VL checkpoint using `scripts/fetch_vl_model.sh`
2. If you haven't already done so, 
    a. Follow the setup instructions `modeling/vl_model/README.md` so that all the depencies for running the VL model are installed.
    b. Run the `modeling/vl_model/download_pretrained.sh` script from inside the `modeling/vl_model` folder. This downloads the pretrained tokenizer and the pytorch JIT compiled CLIP model that is used to initialize the model.
3. Double check the model checkpoint path in `modeling/inference/model_executors/vl_model_executor.py`
4. The model and corresponding executor are defined in `modeling/inference/models/vl_model.py` and `modeling/inference/model_executors/vl_executor.py` respectively.


After the prerequisites for the required model is set up, run the evaluation by running
```
bash run_linux.sh
```

This script sets the paths (modify them as per your need), launches the Xserver and finally spawns the environment and sends it test instructions. This invokes `eval.py` for the chosen model, sets the path for data directories and which instruction set to evaluate upon, while also choosing between the decoding methods of OBJECT_CLASS (language only models with no visual component which directly uses the object ID to navigate in Arena) and OBJECT_MASK (object mask based interaction).

Running the `run_linux.sh` script will spawn the game environment and establish connection to it. It will then read the processed instructions and pass them to the chosen model to generate interaction commands which are then sent to the `arena_orchestrator`. The orchestrator executes these actions on the environment and tracks the mission completion state. Completed missions are tracked and we output the final mission level success rate after iterating over all the missions.

The script will output logs to explain the mission state, the instructions received, the actions generated and mission status.

#### Note
Make sure the game environment is not running before launching the script by doing `top` and making sure no process with the name "Arena" is running. If it is, kill that process with `sudo kill -9 <process_id>` and then launch the script. This often happens when your script crashed in a previous run and you try to run it again, then a previous instance of arena executable might still be running. Make sure to kill it before spawning a new one.
