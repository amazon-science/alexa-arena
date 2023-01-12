# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import os.path
from typing import Any, Callable, Optional, Tuple, List

from PIL import Image
import torchvision.datasets as dset
from preprocessors import get_transform
import torch
import numpy as np
import json
from argparse import ArgumentParser


class CocoDataset(dset.VisionDataset):
    """`MS Coco Detection <https://cocodataset.org/#detection-2016>`_ Dataset.

    It requires the `COCO API to be installed <https://github.com/pdollar/coco/tree/master/PythonAPI>`_.

    Args:
        root (string): Root directory where images are downloaded to.
        annFile (string): Path to json annotation file.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.PILToTensor``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        transforms (callable, optional): A function/transform that takes input sample and its target as entry
            and returns a transformed version.
    """

    def __init__(
        self,
        metadata_file: str,
        args,
        transform: Optional[Callable] = None,
        target_transform: Optional[Callable] = None,
        transforms: Optional[Callable] = None,
    ) -> None:
        if "train" in metadata_file:
            with open(os.path.join(args.coco_data_root, "train", "no_bboxes.json")) as f:
                bad_image_ids = set(json.load(f))
            data_root = os.path.join(args.coco_data_root, "train/data")
        elif "val" in metadata_file:
            with open(os.path.join(args.coco_data_root, "validation", "no_bboxes.json")) as f:
                bad_image_ids = set(json.load(f))
            data_root = os.path.join(args.coco_data_root, "validation/data")
        super().__init__(data_root, transforms, transform, target_transform)
        from pycocotools.coco import COCO

        self.coco = COCO(metadata_file)
        self.ids = set(self.coco.imgs.keys())
        # Images that dont contain any coco annotations, remove them
        self.ids = self.ids - bad_image_ids
        self.ids = sorted(list(self.ids))
        print("Number of images: {}".format(len(self.ids)))

    def _load_image(self, id: int) -> Image.Image:
        path = self.coco.loadImgs(id)[0]["file_name"]
        return Image.open(os.path.join(self.root, path)).convert("RGB")

    def _load_target(self, id: int) -> List[Any]:
        return self.coco.loadAnns(self.coco.getAnnIds(id))

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        id = self.ids[index]
        image = self._load_image(id)
        target = self._load_target(id)

        # Reformatting targets to be compatible with the pipeline
        boxes, labels, masks, areas, iscrowds = [], [], [], [], []
        for tar in target:
            bbox = tar['bbox']
            bbox[2] += bbox[0]
            bbox[3] += bbox[1]
            label = tar['category_id']
            area = tar['area']
            iscrowd = tar['iscrowd']
            mask = self.coco.annToMask(tar)
            boxes.append(bbox)
            labels.append(label)
            masks.append(mask)
            areas.append(area)
            iscrowds.append(iscrowd)

        image_id = torch.tensor([id])
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        areas = torch.as_tensor(areas, dtype=torch.float32) 
        iscrowds = torch.as_tensor(iscrowds, dtype=torch.uint8)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = areas
        target["iscrowd"] = iscrowds
        
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

    def __len__(self) -> int:
        return len(self.ids)


if __name__ == "__main__":
    parser = ArgumentParser()
    coco_data_root = "/home/ubuntu/fiftyone/coco-2017/" 
    parser.add_argument("--coco-train-data-root", dest="coco_train_data_root", type=str,
                        default=os.path.join(coco_data_root, "train/data"),
                        help="Training images folder root for coco dataset.")
    parser.add_argument("--coco-test-data-root", dest="coco_test_data_root", type=str,
                        default=os.path.join(coco_data_root, "validation/data/"),
                        help="Validation/testing images root for coco dataset.")
    parser.add_argument("--coco-train-metadata-file", dest="coco_train_metadata_file", type=str,
                        default=os.path.join(coco_data_root, "raw/instances_train2017.json"),
                        help="Train metadata file for coco dataset.")
    parser.add_argument("--coco-test-metadata-file", dest="coco_test_metadata_file", type=str,
                        default=os.path.join(coco_data_root, "raw/instances_val2017.json"),
                        help="Test metadata root for coco dataset.")
    args = parser.parse_args()
    dataset = CocoDataset(
        metadata_file=args.coco_test_data_root,
        transforms=get_transform(False),
        args=args,
    )
    bad_image_ids = []
    for i in range(len(dataset)):
        print(i)
        if dataset[i][1]['labels'].size()[0] == 0:
            bad_image_ids.append(dataset[i][1]['image_id'].item())
    import pdb; pdb.set_trace()
