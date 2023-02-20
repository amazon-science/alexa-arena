# 1. Installation
This was tested on the Deep Learning AMI (Ubuntu 18.04) Version 61.3 AWS instance.

1. Create a new conda environment and activate it.
    ```
    conda create -n vl_model python=3.7
    conda activate vl_model
    ```

2. Install pytorch
```
pip3 install torch==1.12.1+cu116 torchvision==0.13.1+cu116 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu116
```

3. Install all other reqquirements
```
cd vl_model
pip install -r requirements.txt
```

4. Clone the CLIP repository from https://github.com/openai/CLIP to install the package 
```
git clone https://github.com/openai/CLIP.git
cd CLIP
pip install .
```

## 2. Training
1. Download the trajectory dataset using `scripts/fetch_trajectory_data.sh` and configure the `--images-root` and `data_root` arguments in `train_eval.py`.

2. `cd vl_model` - The following steps need to be run inside the `vl_model` folder. 

3. Download the pretrained clip model and tokenizer by running 
```
sh download_pretrained.sh
```
This saves the pretrained model in `pretrained` folder and the tokenizer in the `data_generators` folder. 

4. For training, we use a batch size of 16 per GPU and train on 4 GPUs (to give a global batchsize of 64)
```
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:1113 --nnodes=1 --nproc_per_node=4 train_eval.py --mode train --exp-num -1 --rank-0-gpu 0 --batch-size 16 --save-log-folder <path/to/save/training/logs>
```
All other arguments are the default arguments specified in `train_eval.py`

5. The trained models are stored in `--save-log-folder`. This can be used for end-to-end mission level evaluation following the instructions in `modeling/inference/README.md`

## 3. Evaluation

Follow the instructions in `modeling/inference/README.md` for doing mission level end-to-end system evaluation with the VL model for dialog guided task completion.
