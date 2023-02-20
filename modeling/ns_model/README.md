## Installation
This was tested on an AWS EC2 p3.8xlarge instance with Ubuntu 18.04. The instance has 4 Tesla V100 GPUs, 32 vCPUs and 244 GB Memory. 

1. Create a new conda environment and activate it.
```
conda create -n ns_model python=3.8
conda activate ns_model
```

2. Install pytorch
```
pip install torch==1.12.1+cu116 torchvision==0.13.1+cu116 --extra-index-url https://download.pytorch.org/whl/cu116
```

3. Install all other requirements
```
pip install -r modeling/ns_model/requirements.txt
```

## Training

1. Run the script to download the trajectory data and the vision model checkpoints.
```
scripts/fetch_vision_model.sh
scripts/fetch_trajectory_data.sh
```

2. The entry point to the training and evaluation code is in `modeling/ns_model/train_eval.py`. The model is an episodic transformer model trained on the training data `data/trajectory-data/train.json`. The evaluation for action prediction accuracy is done on the validation data `data/trajectory-data/valid.json`. 
```
export ALEXA_ARENA_DIR="$HOME/AlexaArena"
CUDA_VISIBLE_DEVICES=0 python -m modeling.ns_model.train_eval
```

3. The trained models are stored in `--checkpt-dir`. By default it is stored in `logs/ns_model_checkpt/`. This can be used for end-to-end mission level evaluation following the instructions in `modeling/inference/README.md`
