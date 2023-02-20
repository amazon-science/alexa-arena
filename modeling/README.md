The modeling baselines are organized into multiple projects (e.g. `modeling/vl_model`), which are then consumed by the `modeling/inference` folder for end-to-end mission level evaluation. The models and executors in the `modeling/inference/models` and `modeling/inference/model+executors` folders are the end-to-end models and execution strategies, respectively, constructed by using the different modeling projects. 

## Model Training & Evaluation

## 1. Placeholder Model
1. The placeholder robot action prediction model does not require training  because it uses a set of heuristic rules to parse the language.
2. To train and evaluate the vision model (Mask R-CNN) stand-alone for mask generation, please follow the [Vision Model README](./vision_model/README.md)
3. For evaluating the end-to-end model for mission completion, follow the [End-to-End Inference README](./inference/README.md) for the placeholder model.

## 2. Vision Language Model
1. To train the VL model for action and mask generation, please follow the [Vision Language Model README](./vl_model/README.md)
2. To evaluate the VL model for mission completion, follow the [End-to-End Inference README](./inference/README.md) for the VL model.

## 3. Neural-Symbolic Model
1. To train the Neural-Symbolic model for action prediction, please follow the [Neural-Symbolic Model README](./ns_model/README.md)
