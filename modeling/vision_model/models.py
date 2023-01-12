# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import numpy as np
import torchvision


# from torchvision.models.detection import MaskRCNN
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection.mask_rcnn import MaskRCNNPredictor, MaskRCNN_ResNet50_FPN_V2_Weights
from torchvision.models.detection.rpn import AnchorGenerator, RPNHead
from torchvision.models.detection.backbone_utils import resnet_fpn_backbone
import torch
import torch.nn as nn
from torchvision import models, transforms
from argparse import ArgumentDefaultsHelpFormatter, ArgumentParser


def get_model_instance_segmentation(num_classes, args):
    model = MaskRCNN(num_classes=num_classes, args=args)
    if args.load_pretrained_model:
        print(f"Loading pretrained model: {args.pretrained_model_path}")
        device_id = args.device_ids[0]
        device = torch.device(f'cuda:{device_id}') if torch.cuda.is_available() else torch.device('cpu')
        checkpoint = torch.load(args.pretrained_model_path, map_location=torch.device(device))["model"]
        # Remove final bbox and mask predictor since this does not match with the number of classes
        if args.dataset == "coco":
            keys_to_delete = set([])
        else:
            keys_to_delete = set([
            "roi_heads.box_predictor.cls_score.weight",
            "roi_heads.box_predictor.cls_score.bias",
            "roi_heads.box_predictor.bbox_pred.weight",
            "roi_heads.box_predictor.bbox_pred.bias",
            "roi_heads.mask_predictor.mask_fcn_logits.weight",
            "roi_heads.mask_predictor.mask_fcn_logits.bias"
        ])
        checkpoint_dict = {k: v for k, v in checkpoint.items() if k not in keys_to_delete}
        # Model weights
        model_dict = model.model.state_dict()
        # Load the pretrained weights
        model_dict.update(checkpoint_dict)
        model.model.load_state_dict(model_dict)

    return model
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # backbone  = resnet_fpn_backbone("resnet18", pretrained=True)
    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                    aspect_ratios=((0.5, 1.0, 2.0),))
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
    #                                                 output_size=7,
    #                                                 sampling_ratio=2)
    # mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    #                                                       output_size=14,
    #                                                       sampling_ratio=2)
    # model = MaskRCNN(backbone=backbone,
    #                  num_classes=num_classes,
    #                  rpn_anchor_generator=anchor_generator,
    #                  box_roi_pool=roi_pooler,
    #                  mask_roi_pool=mask_roi_pooler)
    # return model
    # load an instance segmentation model pre-trained on COCO
    # model = torchvision.models.detection.maskrcnn_resnet50_fpn(pretrained=True)
    # backbone = torchvision.models.resnet18(pretrained=True)
    # backbone.out_channels = 1000
    # model.backbone = backbone
    # anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256, 512),),
    #                                    aspect_ratios=((0.5, 1.0, 2.0),))
    # roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=[0],
    #                                             output_size=7,
    #                                             sampling_ratio=2)
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # # mask_roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],
    # #                                                           output_size=14,
    # #                                                           sampling_ratio=2)
    # hidden_layer = 256
    # # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                    hidden_layer,
    #                                                    num_classes)
    # # model = MaskRCNN(backbone,
    # #                 num_classes=num_classes,
    # #                 rpn_anchor_generator=anchor_generator,
    # #                 box_roi_pool=roi_pooler)
    # #                 # mask_roi_pool=mask_roi_pooler)
    # return model
    # get number of input features for the classifier
    # in_features = model.roi_heads.box_predictor.cls_score.in_features
    # # replace the pre-trained head with a new one
    # model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)

    # # now get the number of input features for the mask classifier
    # in_features_mask = model.roi_heads.mask_predictor.conv5_mask.in_channels
    # hidden_layer = 256
    # # and replace the mask predictor with a new one
    # model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
    #                                                    hidden_layer,
    #                                                    num_classes)

    return model


class Resnet18(object):
    '''
    pretrained Resnet18 from torchvision
    '''

    def __init__(self, eval=True, share_memory=False, use_conv_feat=True, args=None):
        self.model = models.resnet18(pretrained=True)

        if args.gpu:
            self.model = self.model.to(torch.device('cuda'))

        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()

        if use_conv_feat:
            self.model = nn.Sequential(*list(self.model.children())[:-2])

    def extract(self, x):
        return self.model(x)
# (box_predictor): FastRCNNPredictor(
#       (cls_score): Linear(in_features=1024, out_features=119, bias=True)
#       (bbox_pred): Linear(in_features=1024, out_features=476, bias=True)
#     )

class MaskRCNN(object):
    '''
    pretrained MaskRCNN from torchvision
    '''

    def __init__(self, args, eval=True, share_memory=False, min_size=224, num_classes=139, device='cuda'):
        self.model = models.detection.maskrcnn_resnet50_fpn_v2(
            weights=MaskRCNN_ResNet50_FPN_V2_Weights.DEFAULT if args.use_coco_pretrained_model else None,
            min_size=min_size)
        if args.dataset == "arena":
            in_features = self.model.roi_heads.box_predictor.cls_score.in_features
            # replace the pre-trained head with a new one
            self.model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
            # now get the number of input features for the mask classifier
            in_features_mask = self.model.roi_heads.mask_predictor.conv5_mask.in_channels
            hidden_layer = args.hidden_layer_size
            anchor_sizes = ((4,), (16, ), (64,), (128,), (300,))
            anchor_generator = AnchorGenerator(sizes=anchor_sizes,
                                            aspect_ratios=((0.5, 1.0, 2.0), ) * len(anchor_sizes))
            self.model.rpn.anchor_generator = anchor_generator
            self.model.roi_heads.mask_predictor = MaskRCNNPredictor(in_features_mask,
                                                                    hidden_layer,
                                                                    num_classes)
        
        if eval:
            self.model = self.model.eval()

        if share_memory:
            self.model.share_memory()


class Resnet(object):

    def __init__(self, args, eval=True, share_memory=False, use_conv_feat=True):
        self.model_type = args.visual_model
        self.gpu = args.gpu

        # choose model type
        if self.model_type == "maskrcnn":
            self.resnet_model = MaskRCNN(args, eval, share_memory)
        else:
            self.resnet_model = Resnet18(args, eval, share_memory, use_conv_feat=use_conv_feat)

        # normalization transform
        self.transform = self.get_default_transform()


    @staticmethod
    def get_default_transform():
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            )
        ])

    def featurize(self, images, batch=32):
        images_normalized = torch.stack([self.transform(i) for i in images], dim=0)
        if self.gpu:
            images_normalized = images_normalized.to(torch.device('cuda'))

        out = []
        with torch.set_grad_enabled(False):
            for i in range(0, images_normalized.size(0), batch):
                b = images_normalized[i:i+batch]
                out.append(self.resnet_model.extract(b))
        return torch.cat(out, dim=0)

if __name__ == "__main__":
    # parser
    parser = ArgumentParser(formatter_class=ArgumentDefaultsHelpFormatter)
    args = parser.parse_args()
    args.visual_model = "maskrcnn"
    args.gpu = True
    model = MaskRCNN(args)
    # print(model)
    path = "/Users/ssshakia/weight_maskrcnn.pt"
    checkpoint = torch.load(path, map_location=torch.device('cpu'))
    print(len(checkpoint.keys()))
    # print(checkpoint)
    # Remove final bbox and mask predictor since this does not match with the number of classes
    keys_to_delete = set([
        "roi_heads.box_predictor.cls_score.weight",
        "roi_heads.box_predictor.cls_score.bias",
        "roi_heads.box_predictor.bbox_pred.weight",
        "roi_heads.box_predictor.bbox_pred.bias",
        "roi_heads.mask_predictor.mask_fcn_logits.weight",
        "roi_heads.mask_predictor.mask_fcn_logits.bias"
    ])
    keys_to_delete = set([])
    checkpoint_dict = {k: v for k, v in checkpoint.items() if k not in keys_to_delete}
    model_dict = model.model.state_dict()
    model_dict.update(checkpoint_dict)
    model.model.load_state_dict(model_dict)

# size mismatch for roi_heads.box_predictor.cls_score.weight: copying a param with shape torch.Size([119, 1024]) from checkpoint, the shape in current model is torch.Size([159, 1024]).
# size mismatch for roi_heads.box_predictor.cls_score.bias: copying a param with shape torch.Size([119]) from checkpoint, the shape in current model is torch.Size([159]).
# size mismatch for roi_heads.box_predictor.bbox_pred.weight: copying a param with shape torch.Size([476, 1024]) from checkpoint, the shape in current model is torch.Size([636, 1024]).
# size mismatch for roi_heads.box_predictor.bbox_pred.bias: copying a param with shape torch.Size([476]) from checkpoint, the shape in current model is torch.Size([636]).
# size mismatch for roi_heads.mask_predictor.mask_fcn_logits.weight: copying a param with shape torch.Size([119, 256, 1, 1]) from checkpoint, the shape in current model is torch.Size([159, 256, 1, 1]).
# size mismatch for roi_heads.mask_predictor.mask_fcn_logits.bias: copying a param with shape torch.Size([119]) from checkpoint, the shape in current model is torch.Size([159]).
