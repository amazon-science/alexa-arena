## Model Training & Evaluation

## 1. Placeholder Model
1. The placeholder robot action prediction model does not require training  because it uses a set of heuristic rules to parse the language.
2. To train and evaluate the vision model (Mask R-CNN) stand-alone for mask generation, please follow the [Vision Model README](./vision_model/README.md)
3. For evaluating the end-to-end model for mission completion, follow the [End-to-End Inference README](./inference/README.md) for the placeholder model.

## 2. VL Model
1. To train the VL model for action and mask generation, please follow the [Vision Language Model README](./vl_model/README.md)
2. To evaluate the VL model for mission completion, follow the [End-to-End Inference README](./inference/README.md) for the VL model.
