# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


from curses import meta
import os
import numpy as np
import torch
from PIL import Image
import json
from glob import glob
import utils
import transforms as T
import matplotlib.pyplot as plt
from preprocessors import get_transform
from data_cleaning_version_control import area_threshold_v2, area_threshold_v3
from coco_pretrained import map_arena_coco_common
from argparse import ArgumentParser
from data_cleaning_version_control import decode_absent_class_assignments


class ArenaDataset(torch.utils.data.Dataset):
    def __init__(self, metadata_file, transforms=None, args=None):
        self.metadata_file = metadata_file
        self.transforms = transforms
        self.args = args
        with open(self.metadata_file) as f:
            metadata_files = f.readlines()
        self.metadata_files_temp = sorted([os.path.join(args.data_root, line.strip()) for line in metadata_files])
        self.metadata_files = []
        for meta_path in self.metadata_files_temp:
            img_num = meta_path.split("/")[-1].split("_")[0]
            subroot_dir = '/'.join(meta_path.split('/')[:-1])
            if os.path.exists(os.path.join(subroot_dir, "{}_color.png".format(img_num))) and \
                os.path.exists(os.path.join(subroot_dir, "{}_seg.png".format(img_num))) and \
                    os.path.exists(meta_path):
                self.metadata_files.append(meta_path)
        print(f"Number of metadata files in {metadata_file}: {len(self.metadata_files)}")
        with open(self.args.class_to_idx_file) as f:
            self.class2idx = json.load(f)
        # self.class2idx = map_arena_coco_common()
        with open(self.args.objectid_to_class_file) as f:
            self.objectid2class = json.load(f)
        with open(self.args.class_to_objectid_file) as f:
            self.class2objectid = json.load(f)
        with open(self.args.class_to_area_thresholds) as f:
            self.class_to_area_thresholds = json.load(f)
        trajectory_data_classes_file = args.trajectory_data_class_file
        with open(trajectory_data_classes_file) as f:
            self.trajectory_data_class_list = f.readlines()
        self.trajectory_data_class_list = set([cl.strip() for cl in self.trajectory_data_class_list])
        if len(self.trajectory_data_class_list) == 0:
            self.trajectory_data_class_list = set(list(self.class2objectid.keys()))

    def __getitem__(self, idx):
        # Load metadata file
        meta_path = self.metadata_files[idx]
        with open(meta_path) as f:
            metadata = json.load(f)
        # Hack for not adding image file names in the earlier data batches
        img_num = meta_path.split("/")[-1].split("_")[0]
        subroot_dir = '/'.join(meta_path.split('/')[:-1])
        metadata["color_image_file"] = "{}_color.png".format(img_num)
        metadata["segmentation_image_file"] = "{}_seg.png".format(img_num)
        img = self.read_image(os.path.join(subroot_dir, metadata["color_image_file"]))
        seg_mask = self.read_image(os.path.join(subroot_dir, metadata["segmentation_image_file"]))
        boxes = []
        labels = []
        masks = []
        for ann in metadata["image_annotations"]:
            object_id = ann["object_id"]
            #import pdb; pdb.set_trace()
            # To convert the object id in metadata (which may have been appended with instance numbers by RG)
            # and other information added during data generation stage
            class_assigned = self.convert_objectid_to_class(object_id)
            # if class_assigned in self.class2idx:
            class_idx = self.class2idx[class_assigned]
            # import pdb; pdb.set_trace()
            # else:
            #     class_idx = 100 # Randomly assigning a class that doesn't exist in the above mapping for off the shelf model evaluation
            bbox = ann["bbox"]
            # import pdb; pdb.set_trace()
            area = (bbox[3] - bbox[1]) * (bbox[2] - bbox[0])
            # If class is unassigned (class idx of unassigned = 0) and area threshold is met
            try:
                area_threshold = self.class_to_area_thresholds[class_assigned][0]
                area_threshold = min(100, area_threshold)
            except:
                area_threshold = 1
            if class_idx and  area >= area_threshold and class_assigned in self.trajectory_data_class_list:
                boxes.append(bbox)
                labels.append(class_idx)
                rgb = tuple(ann["rgb"])
                mask = np.zeros_like(seg_mask)
                indices = np.where(np.all(seg_mask == rgb, axis=-1))
                mask[indices] = 1
                mask = mask[:, :, 0]
                masks.append(mask)

        image_id = torch.tensor([idx])
        num_objs = len(boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        # import pdb; pdb.set_trace()
        try:
            area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        except:
            area = 0
            print("Found no boxes, cannot calculate areas for {}".format(meta_path))
        labels = torch.as_tensor(np.array(labels), dtype=torch.int64)
        masks = torch.as_tensor(np.array(masks), dtype=torch.uint8)
        iscrowd = torch.zeros((num_objs,), dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["masks"] = masks
        target["image_id"] = image_id
        target["area"] = area
        target["iscrowd"] = iscrowd
        #target["meta_path"] = meta_path
        if self.transforms is not None:
            img, target = self.transforms(img, target)
        return img, target

    def convert_objectid_to_class(self, object_id):
        """
        Utility function to convert the object id in the metadata to barebones
        object id found in the class to object mapping.
        """
        id_split = object_id.split("_")
        object_id_proc = '_'.join(id_split[:-1])
        # object_id_proc = object_id
        if object_id_proc in self.objectid2class:
            class_assignment = self.objectid2class[object_id_proc]
        else:
            class_assignment = decode_absent_class_assignments(
                object_id_proc, object_id, self.objectid2class
            )
            if class_assignment == None:
                class_assignment = "Unassigned"

        if class_assignment not in self.trajectory_data_class_list:
            class_assignment = "Unassigned"
        return class_assignment

    def mask_to_bbox(self, mask):
        # Bounding box.
        rows = np.any(mask, axis=1)
        cols = np.any(mask, axis=0)

        if not np.any(rows) or not np.any(cols):
            return [0, 0, 0, 0]

        rmin, rmax = np.where(rows)[0][[0, -1]]
        cmin, cmax = np.where(cols)[0][[0, -1]]

        return [int(cmin), int(rmin), int(cmax), int(rmax)]

    def __len__(self):
        return len(self.metadata_files)

    def read_image(self, path):
        im_frame = Image.open(path)
        np_frame = np.array(im_frame)[:,:,0:3]  # Leaving out the transperancy channel
        return np_frame
