# 1. Instance set up

This set up instructions can be used for training and evaluation of the vision model.

1. The data is around ~1TB of memory- so make necessary arragements to download and store it.

2. This was tested on the Deep Learning AMI (Ubuntu 18.04) Version 61.3 AWS instance.

### 1.1 Setting up environment

1. `conda create -n simcv python=3.7`
2. `conda activate simcv`
3. Run the following commands to install the dependencies.
```
cd vision_model
pip install -r requirements.txt
pip3 install torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu116
```

# 2. Data

### 2.1 Run fetch_vision_data.sh 
1. The data is organized into four zipped folders `data1.zip`, `data2.zip`, `validation_data.zip` and `object_manifests.zip`.
Run the `scripts/fetch_vision_data.sh` scripts with the paths to all three. This creates a `image_data` folder in the project root folder and downloads the data to it and downloads the object manifests into `data` folder.

2. `data1.zip`, `data2.zip`, and `validation_data.zip` contain all the image files, ground truth segmentations and metadata files.

3. `object_manifests.zip` contains the Arena Objects list and associated properties (see documentation folder of this repository for more info). This file can also be used for converting the object IDs to classes that you can configure according to your needs. Provide the `/local/path` that you download this to as the `--rg-object-list-root` argument in `prepare_clean_data.py`.

4. For more information about the contents of the vision dataset, please refer to `data/vision-data/README.md`

### 2.2 Precomputed artifacts

This repository includes two `class_to_area_thresholds` artifacts. To use this during the data cleaning process, update the `--class-to-area-thresholds-path` in `prepare_clean_data.py`. This is a precomputed dictionary which maps every class to an area pruning threshold, which can be used during cleaning using `prepare_and_clean_data.py` and data generation in `data_generator.py`.
It maps every class to lower and upper area thresholds. These were calculated by mapping every class to the areas of each of their instances from the training data, getting the bottom 10% of the areas and calculating an average for the lower threshold, getting the top 25% of the areas and calculate an average of that for the uppwer threshold. The intuition is that for smaller objects, we can weed out the noise by filtering out very small objects in that class, since the user might need to come closer to the object to be able to detect that object at a reasonable distance. You can implement your own function in the data cleaning or the ML model data generator stage to do such filtering.

### 2. 3.  Prepare and Clean Data

1. Update paths in `prepare_and_clean_data.py`. Specifically, update the `--data-root`,`--rg-object-list-root`. `--data-root` is the parent of the `object_detection_data_v<x>` folders from the data download step. `--rg-object-list-root` is the path to the object manifests folder extracted. 

2. Ensure the `--class-to-area-thresholds-path` is populated with the correct path. 

3. If you want to prepare the data only for a few predefined set of classes, populate the `--classes-path`. This is a list of class names in a text file with each class name occupying a line (do not include the Unassigned class).

4. Run the following 2 commands in series for cleaning and creating training data and validation data. This script can also be modified to do custom cleaning. Currently `data_cleaning_version_control.py` in conjunction with `prepare_and_clean_data.py` has details about how the classes are decided and mapped, collapsed, how annotations are validated and what thresholding is applied. 

```
python prepare_and_clean_data.py --split "train"
python prepare_and_clean_data.py --split "validation"
```

Running the above commands on the entire dataset runs the multiprocessing pipeline in `prepare_and_clean_data.py` (which should take about 1-1.5 hours to run if running on the entire dataset). At the end, it will give the total number of classes "Number of readable names / classes for the model" which should be updated in `train_eval.py` in `--num-classes` (under Data args) before starting to train. It also prints out a bunch of other useful information.

The data entry point text files are now stored in a folder within the `modeling/vision_model` folder of the format `datav<x>_collapsev<x>_isvalidv<x>_rgv<x.y>_<split>_<date_month_year>`. This also saves the class frequencies, object to class and class to object mappings in the same folder.  


# 4. Training and Evaluation

### 4.1 Training

1. The entry point to the training code is in `train_eval.py`. 
2. Update the data paths in the arguments. Update `rel_data_dir_train`, `rel_data_dir_test` in `train_eval.py`
3. For single and multigpu distributed training, where --nproc_per_node is the number of GPUs to use on your node. 
```
torchrun --nproc_per_node=2 train_eval.py  
```
If there are multiple training jobs to be run on the same instance:
```
torchrun --rdzv_backend=c10d --rdzv_endpoint=localhost:54321 --nnodes=1 --nproc_per_node=2 train_eval.py 
```
Please see https://pytorch.org/docs/stable/elastic/run.html for more information.

Please note that Mask R-CNN in pytorch is not compatible with data parallel mode of training. However, the SOTA Mask R-CNN models use a batch size of 16 which can be trained on only one Tesla V100 gpu. Having said that, if you are implementing other Faster R-CNN based models that require more GPUs, please resort to the distributed training procedure above (distributed training command can be used even for a single node single gpu training by setting --nnodes=1 --nproc_per_node=1). 

4. The code can be run in `train` mode which does only training, `train_evaluate` mode which trains and evaluates every `--eval-freq` epochs, and `evaluate` mode which evaluates a given experiment's checkpoint. The evaluation checkpoint in `train_evaluate` mode is overridden during runtime to the checkpoint being evaluated during training.

5. Please update the arguments in `train_eval.py` before starting to train. All training logs will be stored in `--save-log-folder`. Update this argument as required.

6. Please read the argument descriptions in `train_eval.py` for more details. All descriptions can be read from the script. The configs are divided as follows (these are separated out in the args in `train_eval.py`):
    - Running mode.
    - Training devices
    - Experiment details
    - Data
    - Model
    - Hyperparameters
    - Logs
    - Evaluation arguments


### 4.2 Placeholder model

Please refer to `fetch_vision_model.sh` to download the placeholder vision model. The placeholder model was trained on 1 Tesla V100 gpu with a batch size of 16 on all data folders except `object_detection_data_v1` (this folder contains relatively few images and was left out).


### 4.3 Evaluation
A trained model can be evaluated during training by setting the `--mode` to `train_evaluate` or can be evaluated standalone in the `evaluate` mode. 

For standalone evaluation, either `--eval-exp-num` and `--eval-checkpoint` args can be used to load the model checkpoint from the training logs and store all metrics in a folder inside the same logs folder.

If not loading the model from the training logs, standalone evaluation can also be done by providing a model checkpoint path using `--eval-checkpoint-path-custom` (path to the pytorch checkpoint) and `--eval-save-dir-custom` (path where you want to store the metrics and outputs of the evaluation). Update the `--mode` in `train_eval.py` to `evaluate` and provide the path to the model in `--eval-checkpoint-path-custom` and the directory to save metrics and other evaluation artifacts to at `--eval-save-dir-custom`. These need not be the same folder.

To run evaluation on one GPU:

```
torchrun --nproc_per_node=1 train_eval.py --mode evaluate --eval-exp-num <x> --eval-checkpoint <y>
```

After the evaluation has completed running, the metrics and other evaluation artifacts can be inspected in the `--eval-save-dir-custom` location or the `eval` folder in the respective training logs folder, depending on how you ran the standalone evaluation. An itemized list of artifacts and what they signify is described below:

1. `area_0_1296_classwise_score_iou_map.json`: class-wise mAP for all different (score, iou) thresholds for objects with small area (0 > area > 1296)
2. `area_0_1296_mAP_mAR_curve.json`: all classes mAP for all different (score, iou) thresholds for objects with small area (0 > area >= 1296)
3. `area_1296_9216_classwise_score_iou_map.json`: class-wise mAP for all different (score, iou) thresholds for objects with medium area (1296 > area >= 9216) 
4. `area_1296_9216_mAP_mAR_curve.json`: all classes mAP for all different (score, iou) thresholds for objects with medium area (1296 > area >= 9216)
5. `area_9216_90000_classwise_score_iou_map.json`: class-wise mAP for all different (score, iou) thresholds for objects with large area (9216 > area >= 90000)
6. `area_9216_90000_mAP_mAR_curve.json`: all classes mAP for all different (score, iou) thresholds for objects with large area (9216 > area >= 90000)
7. ROC curves: `<area_category>_roc_curve.png` -  visualizing the ROC curves for mAP for various score iou thresholds. 
8. `confusion_matrices/area_<area_category>` - all confusion matrix numpy arrays and images for all score iou thresholds.
9. `uber_areawise_map.json` - Averages all the (score, iou) mAPs across all classes.

In addition to the above (score, threshold) operating point wise metrics, we also provide the scripts for calculating the mAP as defined in the Mask RCNN paper, i.e instead of thresholding on the score, they are thresholded on the number of predictions. This can be saved and analyzed by activating the `--save-coco-metrics` and `--save-cat-metrics-to-disk` flags in `train_eval.py`.
The results for the model are provided in the paper.

#### Possible errors and debugging

The evaluation code stores a coco dataset cache during evaluation for speeding up evaluation. A warning is printed out on the console during the evaluation run. This can be found in `coco_utils.py` in the `convert_to_coco_api` method. This is helpful during large dataset evaluations which could take hours to preprocess the dataset into the coco format. This also prints out a warning saying that the cache is stored and loaded in subsequent runs. If the stored dataset is different from the dataset being evaluated during that run, especially if the cached dataset doesn't have any data that you are trying to evaluate for in the current run, it will throw this error:

`AssertionError: Results do not correspond to current coco set`

The way to fix this error is to remove the pickle file, `rm -rf dataset_small.pickle` from the `vision_model` folder and rerun evaluation. This will again store a cache dataset which will be loaded in subsequent runs.

If the dataset being evaluated in the current run is a subset of the cached dataset, then it will not throw the error.
