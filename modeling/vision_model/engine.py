# Modified from https://github.com/pytorch/vision/blob/main/references/detection/engine.py- Licensed under the
# BSD-3-Clause License.
#
# Modifications Copyright 2022 Amazon.com, Inc. or its affiliates. All Rights Reserved.


import math
import sys
import time
import torch
import os
import matplotlib.pyplot as plt
import numpy as np
import json
import torchvision.models.detection.mask_rcnn
import itertools
from coco_utils import get_coco_api_from_dataset
from coco_eval import CocoEvaluator
import utils
from coco_pretrained import map_arena_coco_common


import resource
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8000, rlimit[1]))

def train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq, scaler=None):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('lr', utils.SmoothedValue(window_size=1, fmt='{value:.6f}'))
    header = 'Epoch: [{}]'.format(epoch)

    lr_scheduler = None
    if epoch == 0:
        warmup_factor = 1. / 1000
        warmup_iters = min(1000, len(data_loader) - 1)

        lr_scheduler = utils.warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor)
    for images, targets in metric_logger.log_every(data_loader, print_freq, header):
        images = list(image.to(device) for image in images)
        targets = [{k: v.to(device) for k, v in t.items()}  for t in targets] 
        with torch.cuda.amp.autocast(enabled=scaler is not None):
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        # reduce losses over all GPUs for logging purposes
        loss_dict_reduced = utils.reduce_dict(loss_dict)
        losses_reduced = sum(loss for loss in loss_dict_reduced.values())

        loss_value = losses_reduced.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            print(loss_dict_reduced)
            sys.exit(1)

        optimizer.zero_grad()
        losses.backward()
        optimizer.step()

        if lr_scheduler is not None:
            lr_scheduler.step()

        metric_logger.update(loss=losses_reduced, **loss_dict_reduced)
        metric_logger.update(lr=optimizer.param_groups[0]["lr"])

    return metric_logger


def _get_iou_types(model):
    model_without_ddp = model
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_without_ddp = model.module
    iou_types = ["bbox"]
    if isinstance(model_without_ddp, torchvision.models.detection.MaskRCNN):
        iou_types.append("segm")
    if isinstance(model_without_ddp, torchvision.models.detection.KeypointRCNN):
        iou_types.append("keypoints")
    return iou_types


@torch.no_grad()
def evaluate(model, data_loader, device, args):
    if args.evaluate_coco_pretrained:
       print(f"\nBeginning evaluation of coco pretrained model. Saving metrics to {args.eval_save_dir_custom}")
    elif args.eval_checkpoint_path_custom:
        print(f"\nBeginning evaluation of model {args.eval_checkpoint_path_custom}")
    else:
        print(f"\nBeginning evaluation of exp num {args.eval_exp_num}, epoch {args.eval_checkpoint}")
    path_to_save = f"{args.save_log_folder}/exp{args.eval_exp_num}/eval/{args.eval_checkpoint}"
    if args.eval_checkpoint_path_custom or args.eval_save_dir_custom:
        path_to_save = args.eval_save_dir_custom
    if not os.path.exists(path_to_save):
        os.makedirs(path_to_save)
    with open(os.path.join(path_to_save, f"parameters_{args.mode}.json"), 'w') as f:
        json.dump(args.__dict__, f, indent=2)
    n_threads = torch.get_num_threads()
    # FIXME remove this and make paste_masks_in_image run on the GPU
    torch.set_num_threads(1)
    cpu_device = torch.device("cpu")
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Test:'

    coco = get_coco_api_from_dataset(data_loader.dataset)
    iou_types = _get_iou_types(model)
    coco_evaluator = CocoEvaluator(coco, iou_types)

    # Confusion matrix and AP/AR calc
    path_to_save = f"{args.save_log_folder}/exp{args.eval_exp_num}/eval/{args.eval_checkpoint}" 
    class_to_idx_path = args.class_to_idx_file
    with open(class_to_idx_path) as f:
        class2idx = json.load(f)
    idx2class = {v: k for k, v in class2idx.items()}
    score_thresholds = [0.05, 0.1, 0.3, 0.5, 0.7]
    iou_thresholds = [0.1, 0.3, 0.4, 0.5, 0.75, 0.8]
    score_iou_thresholds = list(itertools.product(score_thresholds, iou_thresholds))
    area_thresholds = [(0, 36 * 36), (36 * 36, 96 * 96), (96 * 96, 300 * 300)]
    # scores_ious, mAPs, mARs = [], [], []
    # score_iou_threshold_to_mAP_mAR = {}
    # score_iou_threshold_to_class_wise_map = {}
    area_2_score_iou_2_class_2_tp_fp = {
            lower_upper_area: {
            tup: {k: {'tp': 0, 'fp': 0, 'fn': 0} for k, _ in class2idx.items()} for tup in score_iou_thresholds
            }
            for lower_upper_area in area_thresholds
        }
    gt_row_pred_col = np.zeros((len(idx2class), len(idx2class)))
    area_2_score_iou_2_gt_row_pred_col = {
        lower_upper_area: {
        tup: gt_row_pred_col for tup in score_iou_thresholds 
        } 
        for lower_upper_area in area_thresholds
    }

    for images, targets in metric_logger.log_every(data_loader, 100, header):
        images = list(img.to(device) for img in images)
        torch.cuda.synchronize()
        model_time = time.time()
        outputs = model(images)
        outputs = [{k: v.to(cpu_device) for k, v in t.items()} for t in outputs]
        model_time = time.time() - model_time

        res = {target["image_id"].item(): output for target, output in zip(targets, outputs)}
        if args.save_images_masks_to_disk:
            save_images_masks(images=images, targets=targets, res=res, args=args)
        evaluator_time = time.time()
        coco_evaluator.update(res)
        if args.dataset == "arena":
            aggregate_gts_preds(targets, res, score_iou_thresholds, area_thresholds, area_2_score_iou_2_class_2_tp_fp, area_2_score_iou_2_gt_row_pred_col, idx2class, args)
        # import pdb; pdb.set_trace()
        evaluator_time = time.time() - evaluator_time
        metric_logger.update(model_time=model_time, evaluator_time=evaluator_time)
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    coco_evaluator.synchronize_between_processes()
    # accumulate predictions from all images
    coco_evaluator.accumulate()
    coco_evaluator.summarize()
    if args.save_coco_metrics:
        save_metrics(coco_evaluator.coco_eval, args)
    if args.dataset == "arena":
        compute_confusion_matrices_and_metrics(area_2_score_iou_2_class_2_tp_fp, area_2_score_iou_2_gt_row_pred_col, args)
    torch.set_num_threads(n_threads)
    if args.evaluate_coco_pretrained:
        print("\nEvaluated coco pretrained pytorch model")
    elif args.eval_checkpoint_path_custom:
        print(f"\nEvaluated model {args.eval_checkpoint_path_custom}") 
    else: 
        print(f"\nEvaluated exp num {args.eval_exp_num}, epoch {args.eval_checkpoint}\n")
    return coco_evaluator

def save_metrics(eval_metrics, args):
    if args.eval_checkpoint_path_custom or args.eval_save_dir_custom:
        path_to_save = os.path.join(args.eval_save_dir_custom, 'coco_metrics.json')
    else:
        path_to_save = f"{args.save_log_folder}/exp{args.eval_exp_num}/eval/{args.eval_checkpoint}/coco_metrics.json"
    iou_type_to_metrics = {}
    eval_metrics_as_defined_in_pycoco = [
        "Average Precision, areaRng = all, iouThr = 0.5:0.95, maxDets = 100",
        "Average Precision, areaRng = all, iouThr = 0.5, maxDets = 100",
        "Average Precision, areaRng = all, iouThr = 0.75, maxDets = 100",
        "Average Precision, areaRng = small, iouThr = 0.5:0.95, maxDets = 100",
        "Average Precision, areaRng = medium, iouThr = 0.5:0.95, maxDets = 100",
        "Average Precision, areaRng = large, iouThr = 0.5:0.95, maxDets = 100",
        "Average Recall, areaRng = all, iouThr = 0.5:0.95, maxDets = 1",
        "Average Recall, areaRng = all, iouThr = 0.5, maxDets = 10",
        "Average Recall, areaRng = all, iouThr = 0.75, maxDets = 100",
        "Average Recall, areaRng = small, iouThr = 0.5:0.95, maxDets = 100",
        "Average Recall, areaRng = medium, iouThr = 0.5:0.95, maxDets = 100",
        "Average Recall, areaRng = large, iouThr = 0.5:0.95, maxDets = 100",
        ]
    for iou_type in eval_metrics:
        iou_type_to_metrics[iou_type] = {}
        for i, metric in enumerate(eval_metrics_as_defined_in_pycoco):
            iou_type_to_metrics[iou_type][metric] = eval_metrics[iou_type].stats[i]
    with open(path_to_save, 'w') as f:
        json.dump(iou_type_to_metrics, f, indent=2)

def compute_confusion_matrices_and_metrics(area_2_score_iou_2_class_2_tp_fp, area_2_score_iou_2_gt_row_pred_col, args):
    if args.eval_checkpoint_path_custom or args.eval_save_dir_custom:
        path_to_save = args.eval_save_dir_custom
    else:
        path_to_save = f"{args.save_log_folder}/exp{args.eval_exp_num}/eval/{args.eval_checkpoint}"
    confusion_matrices_folder = os.path.join(path_to_save, 'confusion_matrices')
    if not os.path.exists(confusion_matrices_folder):
        os.makedirs(confusion_matrices_folder)
    json_friendly_score_iou_2_class_2_tp_fp = {}
    confusion_matrices_folder = os.path.join(path_to_save, 'confusion_matrices')
    if not os.path.exists(confusion_matrices_folder):
        os.makedirs(confusion_matrices_folder) 
    score_iou_threshold_to_mAP_mAR = {}
    areawise_uber_map = {}
    for area_cat in area_2_score_iou_2_class_2_tp_fp:
        scores_ious, mAPs, mARs = [], [], []
        lower_threshold, upper_threshold = area_cat[0], area_cat[1]
        confusion_area_folder = os.path.join(confusion_matrices_folder, f"area_{lower_threshold}_{upper_threshold}")
        if not os.path.exists(confusion_area_folder):
            os.makedirs(confusion_area_folder)  
        for score_threshold, iou_threshold in area_2_score_iou_2_class_2_tp_fp[area_cat]:
            TP_AGGREGATE, FP_AGGREGATE, FN_AGGREGATE = 0, 0, 0
            json_friendly_score_iou_2_class_2_tp_fp[str((score_threshold, iou_threshold))] = area_2_score_iou_2_class_2_tp_fp[area_cat][(score_threshold, iou_threshold)] 
            for class_label in area_2_score_iou_2_class_2_tp_fp[area_cat][(score_threshold, iou_threshold)]:
                try:
                    tp = area_2_score_iou_2_class_2_tp_fp[area_cat][(score_threshold, iou_threshold)][class_label]["tp"]
                    fp = area_2_score_iou_2_class_2_tp_fp[area_cat][(score_threshold, iou_threshold)][class_label]["fp"]
                    fn =  area_2_score_iou_2_class_2_tp_fp[area_cat][(score_threshold, iou_threshold)][class_label]["fn"]
                    TP_AGGREGATE += tp
                    FP_AGGREGATE += fp
                    FN_AGGREGATE += fn
                    ap =  tp / (tp + fp)
                    area_2_score_iou_2_class_2_tp_fp[area_cat][(score_threshold, iou_threshold)][class_label]["ap"] = ap
                    ar = tp / (tp + fn)
                    area_2_score_iou_2_class_2_tp_fp[area_cat][(score_threshold, iou_threshold)][class_label]["ar"] = ar 
                except ZeroDivisionError:
                    continue
            scores_ious.append(str((score_threshold, iou_threshold)))
            try:
                mAP = TP_AGGREGATE / (TP_AGGREGATE + FP_AGGREGATE)
                mAR = TP_AGGREGATE / (TP_AGGREGATE + FN_AGGREGATE)
            # If ground truth does not exist
            except ZeroDivisionError:
                mAP, mAR = 0, 0
            mAPs.append(mAP)
            mARs.append(mAR)
            score_iou_threshold_to_mAP_mAR[str((score_threshold, iou_threshold))] = (mAP, mAR)
            confusion_matrix = area_2_score_iou_2_gt_row_pred_col[area_cat][(score_threshold, iou_threshold)]
            plt.imshow(confusion_matrix)
            confusion_matrix_img_path = os.path.join(
                confusion_area_folder, 
                f'confusion_matrix_score_{score_threshold}_iou{iou_threshold}_area{lower_threshold}_{upper_threshold}.png'
            )
            plt.savefig(confusion_matrix_img_path)
            plt.close()
            print(f'Saving confusion matrix image to {confusion_matrix_img_path}')
            confusion_matrix_numpy_path = os.path.join(
                confusion_area_folder, 
                f'confusion_matrix_score_{score_threshold}_iou{iou_threshold}_area{lower_threshold}_{upper_threshold}.npy'
            )
            np.save(confusion_matrix_numpy_path, confusion_matrix)
            print(f'Saving confusion matrix numpy to {confusion_matrix_numpy_path}')

        json_friendly_score_iou_2_class_2_tp_fp_path = os.path.join(
            path_to_save, f'area_{lower_threshold}_{upper_threshold}_classwise_score_iou_map.json')
        with open(json_friendly_score_iou_2_class_2_tp_fp_path, 'wt') as f:
            json.dump(json_friendly_score_iou_2_class_2_tp_fp, f, indent=2) 
        map_mar_curve_json_path = os.path.join(path_to_save, f'area_{lower_threshold}_{upper_threshold}_mAP_mAR_curve.json')
        all_maps = [score_iou_threshold_to_mAP_mAR[key][0] for key in score_iou_threshold_to_mAP_mAR.keys()]
        uber_mAP = sum(all_maps) / len(all_maps)
        areawise_uber_map[str(area_cat)] = uber_mAP

        with open(map_mar_curve_json_path, 'wt') as f:
            json.dump(score_iou_threshold_to_mAP_mAR, f, indent=2)
        print(f'Saving mAP-mAR curve json to {map_mar_curve_json_path}')
        roc_curve_plot_path = os.path.join(path_to_save, f'area_{lower_threshold}_{upper_threshold}_roc_curve.png')
        xs = [i + 1 for i in range(len(scores_ious))]
        plt.plot(xs, mAPs, color='r')
        plt.plot(xs, mARs, color='b')
        plt.legend(['mAP', 'mAR'])
        plt.tight_layout()
        plt.xticks(xs, scores_ious, rotation ='vertical')
        plt.title("mAP-mAR vs (score_threshold, iou_threshold)")
        plt.xlabel("(score_threshold, iou_threshold)")
        plt.ylabel("mAP / mAR")
        plt.grid()
        plt.savefig(roc_curve_plot_path, bbox_inches='tight')
        plt.close()
        print(f'Saving roc curve plot to {roc_curve_plot_path}')
    print('\n\nAreawise Uber-mAP: \n')
    print(areawise_uber_map)
    uber_map_path = os.path.join(path_to_save, f'uber_areawise_map.json')
    with open(uber_map_path, 'wt') as f:
        json.dump(areawise_uber_map, f, indent=2)

def aggregate_gts_preds(
    targets, res, score_iou_thresholds, area_thresholds,
    area_2_score_iou_2_class_2_tp_fp, area_2_score_iou_2_gt_row_pred_col,
    idx2class, args
    ):
    for (score_threshold, iou_threshold) in score_iou_thresholds:
        for i in range(len(targets)):
            img_id = targets[i]["image_id"].item()
            score_thresholded_pred_box_idxs = torch.where(res[img_id]['scores'] >= score_threshold)[0]
            pred_boxes = res[img_id]['boxes'][score_thresholded_pred_box_idxs].cpu()
            pred_labels = res[img_id]["labels"][score_thresholded_pred_box_idxs].cpu()
            gt_labels = targets[i]["labels"]
            gt_boxes = targets[i]['boxes']
            target_boxes_expanded = torch.unsqueeze(gt_boxes, 0).repeat(pred_labels.shape[0], 1, 1)
            pred_boxes_expanded = torch.unsqueeze(pred_boxes, 1).repeat(1, gt_labels.shape[0], 1)
            
            intersect_top_left = torch.max(target_boxes_expanded[:, :, 0:2], pred_boxes_expanded[:, :, 0:2])
            intersect_bottom_right = torch.min(target_boxes_expanded[:, :, 2:4],pred_boxes_expanded[:, :, 2:4])
            intersect_length_width = torch.max(
                torch.tensor([0.0]), intersect_bottom_right - intersect_top_left) + torch.tensor([1.0])
            intersect_areas = intersect_length_width[:, :, 0] * intersect_length_width[:, :, 1]
            target_lengths = target_boxes_expanded[:, :, 2] - target_boxes_expanded[:, :, 0] + torch.tensor([1.0])
            target_widths = target_boxes_expanded[:, :, 3] - target_boxes_expanded[:, :, 1] + torch.tensor([1.0])
            target_areas = target_lengths * target_widths

            pred_lengths = pred_boxes_expanded[:, :, 2] - pred_boxes_expanded[:, :, 0] + torch.tensor([1.0])
            pred_widths = pred_boxes_expanded[:, :, 3] - pred_boxes_expanded[:, :, 1] + torch.tensor([1.0])
            pred_areas = pred_lengths * pred_widths 
            iou_tensor = intersect_areas / (target_areas + pred_areas - intersect_areas)
            try:
                pred_bbox_to_gt_idx = torch.argmax(iou_tensor, dim=1)
            except:
                continue
            tp, fp = 0, 0
            duplicate_gt_fp_tracking = set()
            for m in range(pred_labels.shape[0]):
                gt_bbox_idx = pred_bbox_to_gt_idx[m].item() 
                pred_class_idx = pred_labels[m].item()
                gt_class_idx = gt_labels[gt_bbox_idx].item()
                pred_label = idx2class[pred_class_idx]
                gt_box = gt_boxes[gt_bbox_idx]
                area = (gt_box[3] - gt_box[1]) * (gt_box[2] - gt_box[0])
                if area >= 0 and area < (36 * 36):
                    area_cat = (0, 36 * 36)
                elif area >= (36 * 36) and area < (96 * 96):
                    area_cat = (36 * 36, 96 * 96)
                else:
                    area_cat = (96 * 96, 300 * 300)
                        
                if pred_class_idx == gt_class_idx:
                    if iou_tensor[m, gt_bbox_idx] >= iou_threshold:
                        if gt_bbox_idx not in duplicate_gt_fp_tracking:
                            tp += 1
                            area_2_score_iou_2_class_2_tp_fp[area_cat][(score_threshold, iou_threshold)][pred_label]["tp"] += 1
                            duplicate_gt_fp_tracking.add(gt_bbox_idx)
                            area_2_score_iou_2_gt_row_pred_col[area_cat][(score_threshold, iou_threshold)][gt_class_idx, pred_class_idx] += 1
                        else:
                            fp += 1
                            area_2_score_iou_2_class_2_tp_fp[area_cat][(score_threshold, iou_threshold)][pred_label]["fp"] += 1
                else:
                    fp += 1
                    area_2_score_iou_2_class_2_tp_fp[area_cat][(score_threshold, iou_threshold)][pred_label]["fp"] += 1
                    area_2_score_iou_2_gt_row_pred_col[area_cat][(score_threshold, iou_threshold)][gt_class_idx, pred_class_idx] += 1
            for gt_bbox_idx, gt_idx in enumerate(gt_labels):
                if gt_bbox_idx not in duplicate_gt_fp_tracking:
                    gt_label = idx2class[gt_idx.item()]
                    gt_box = gt_boxes[gt_bbox_idx]
                    area = (gt_box[3] - gt_box[1]) * (gt_box[2] - gt_box[0])
                    if area >= 0 and area < (36 * 36):
                        area_cat = (0, 36 * 36)
                    elif area >= (36 * 36) and area < (96 * 96):
                        area_cat = (36 * 36, 96 * 96)
                    else:
                        area_cat = (96 * 96, 300 * 300)
                    area_2_score_iou_2_class_2_tp_fp[area_cat][(score_threshold, iou_threshold)][gt_label]["fn"] += 1


def save_images_masks(images, targets, res, args):
    if args.eval_checkpoint_path_custom or args.eval_save_dir_custom:
        path_to_save = args.eval_save_dir_custom
    else: 
        path_to_save = f"{args.save_log_folder}/exp{args.eval_exp_num}/eval/{args.eval_checkpoint}"
    # path_to_save = "off_the_shelf"
    class_to_idx_path = args.class_to_idx_file
    with open(class_to_idx_path) as f:
        class2idx = json.load(f)
    idx2class = {v: k for k, v in class2idx.items()}
    for i in range(len(images)):
        img_id = targets[i]["image_id"].item()
        img_dir = os.path.join(path_to_save, str(img_id))
        os.makedirs(img_dir)
        gt_path = os.path.join(img_dir, f"{i}.png")
        img = images[i].permute(1, 2, 0).cpu()
        plt.imshow(img)
        plt.savefig(gt_path)
        plt.close()
        pred_mask_dir = os.path.join(img_dir, "pred_masks")
        os.makedirs(pred_mask_dir)
        pred_masks = res[img_id]["masks"].cpu()
        pred_labels = res[img_id]["labels"].cpu()
        scores = res[img_id]["scores"].cpu()
        for m in range(pred_masks.shape[0]):
            pred_label = idx2class[pred_labels[m].item()]
            pred_score = scores[m].item()
            plt.imshow(pred_masks[m][0])
            plt.title(f"Label: {pred_label}, Score: {pred_score}")
            plt.savefig(os.path.join(pred_mask_dir, f"{m}.png"))
            plt.close()
        gt_mask_dir = os.path.join(img_dir, "gt_masks")
        os.makedirs(gt_mask_dir)
        gt_labels = targets[i]["labels"]
        gt_masks = targets[i]["masks"]
        for m in range(gt_masks.shape[0]):
            gt_label = idx2class[gt_labels[m].item()]
            plt.imshow(gt_masks[m])
            plt.title(f"GT Label: {gt_label}")
            plt.savefig(os.path.join(gt_mask_dir, f"{m}.png"))
            plt.close()
