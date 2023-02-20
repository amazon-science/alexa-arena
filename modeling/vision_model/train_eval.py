# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1


import json
import os
from argparse import ArgumentParser

import numpy as np
import torch

import utils
from data_generators.arena_data_generator import ArenaDataset
from data_generators.coco_data_generator import CocoDataset
from engine import evaluate, train_one_epoch
from models import get_model_instance_segmentation
from preprocessors import get_transform
import shutil
import resource
import constants
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8000, rlimit[1]))


def adjust_lr(optimizer, init_lr, epoch, decay_epoch=15, decay_factor=0.1):
    '''
    decay learning rate every decay_epoch
    '''
    lr = init_lr * (decay_factor ** (epoch // decay_epoch))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return optimizer


def load_model_checkpoint(model, path_to_model_checkpoint):
    checkpoint = torch.load(path_to_model_checkpoint)
    model.model.load_state_dict(checkpoint["model"])
    return model

def main(args):
    utils.init_distributed_mode(args)
    num_classes = args.num_classes
    training = True if (args.mode in {"train", "train_evaluate"}) else False
    # use our dataset and defined transformations
    if training:
        if args.dataset == "coco":
            dataset = CocoDataset(
                metadata_file=args.coco_train_metadata_file,
                transforms=get_transform(True),
                args=args
            )
        elif args.dataset == "arena":
            dataset = ArenaDataset(metadata_file=args.train_metadata_file, transforms=get_transform(True), args=args)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset)
        train_batch_sampler = torch.utils.data.BatchSampler(train_sampler, args.batch_size, drop_last=True)
        data_loader = torch.utils.data.DataLoader(
            dataset, batch_sampler=train_batch_sampler, num_workers=args.num_workers,
            collate_fn=utils.collate_fn)
        #data_loader = torch.utils.data.DataLoader(
        #    dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        #    collate_fn=utils.collate_fn)
    if args.mode in {"evaluate", "train_evaluate"}:
        if args.dataset == "coco":
            dataset_test = CocoDataset(
                metadata_file=args.coco_test_metadata_file,
                transforms=get_transform(False),
                args=args
            )
        elif args.dataset == "arena":
            dataset_test = ArenaDataset(metadata_file=args.test_metadata_file, transforms=get_transform(False), args=args)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test, batch_size=16, sampler=test_sampler, num_workers=args.num_workers, collate_fn=utils.collate_fn)

    # get the model using our helper function
    model = get_model_instance_segmentation(num_classes=num_classes, args=args)
    print("Model loaded!")
    model = model.model
    device = "cuda"
    model.to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in params])
    print("Number of model params: ", num_params)
    if args.distributed:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
        model_without_ddp = model.module
    # construct an optimizer
    lr = args.lr
    decay_epoch = args.decay_epoch
    decay_factor = args.decay_factor
    params = [p for p in model.parameters() if p.requires_grad]
    if args.optimizer == "SGD":
        optimizer = torch.optim.SGD(params, lr=lr,
                                    momentum=args.momentum, weight_decay=args.weight_decay)
    elif args.optimizer == "Adam":
        optimizer = torch.optim.Adam(params, lr=lr, weight_decay=args.weight_decay)
    else:
        raise ValueError("Optimizer can't be {}".format(args.optimizer))

    exp_save_log_folder = os.path.join(args.save_log_folder, f"exp{args.exp_num}")
    if utils.is_main_process():
        if not os.path.exists(exp_save_log_folder):
            os.makedirs(exp_save_log_folder)
            os.makedirs(os.path.join(exp_save_log_folder, "model_checkpoints"))
        elif training and not args.resume:
            raise ValueError(f"Exp folder already exists at {exp_save_log_folder},"
                            " Please delete / change exp num / ensure there is no unwanted overwriting of information.")
        if not args.resume and args.mode != "evaluate":
            src_save_folder = os.path.join(exp_save_log_folder, "src")
            if not os.path.exists(src_save_folder):
                os.makedirs(src_save_folder)
                shutil.make_archive(
                    f"{src_save_folder}/src", 'zip',
                    f"{args.project_root}/")
            print(f"Saving source code to {src_save_folder}/src.zip")
            with open(os.path.join(exp_save_log_folder, f"parameters_{args.mode}.json"), 'w') as f:
                json.dump(args.__dict__, f, indent=2)
        stats_save_path = os.path.join(exp_save_log_folder, "stats_logs")
        if not os.path.exists(stats_save_path):
            os.makedirs(stats_save_path)
    if training:
        num_epochs = args.num_epochs
        start_epoch = 0
        if args.resume:   
            resume_checkpoint = utils.get_last_saved_checkpoint(exp_save_log_folder)
            print(f"Resuming from checkpoint {resume_checkpoint}")
            start_epoch = resume_checkpoint + 1
            path_to_model_checkpoint = os.path.join(exp_save_log_folder, f"model_checkpoints/{resume_checkpoint}.pth")
            checkpoint = torch.load(path_to_model_checkpoint,  map_location="cpu")
            state_dict = checkpoint['model'] 
            try:
                model_without_ddp.load_state_dict(state_dict)            
            except:
                state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
                model_without_ddp.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint["optim"])

        print(f"Launching training experiment {args.exp_num}!")
        for epoch in range(start_epoch, num_epochs):
            optimizer.zero_grad()
            print(f"\nEpoch {epoch}")
            # train for one epoch, printing every 10 iterations
            metric_logger = train_one_epoch(model, optimizer, data_loader, device, epoch, print_freq=10, scaler=None)
            # update the learning rate
            optimizer = adjust_lr(optimizer, lr, epoch, decay_epoch=decay_epoch, decay_factor=decay_factor)
            if utils.is_main_process():
                fsave = os.path.join(exp_save_log_folder, "model_checkpoints", '{}.pth'.format(epoch))
                stats = {sv: metric_logger.meters[sv].value for sv in metric_logger.meters}
                torch.save({
                    'metric': stats,
                    'model': model_without_ddp.state_dict(),
                    'optim': optimizer.state_dict(),
                    'epoch': epoch,
                    'args': args
                    }, fsave)
                fbest = os.path.join(stats_save_path, f"{epoch}.json")
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)
                # evaluate on the test dataset
            if (args.mode == "train_evaluate") and ((epoch + 1) % args.eval_freq == 0):
                args.eval_checkpoint = epoch
                args.eval_exp_num = args.exp_num
                if args.save_onnx_during_eval:
                    onnx_save_path = os.path.join(exp_save_log_folder, "model_checkpoints", f"{epoch}.onnx")
                    utils.save_onnx_model(model.model, device, onnx_save_path)
                evaluate(model, data_loader_test, device=device, args=args)
                torch.cuda.empty_cache()
        print("Training complete.")
    else:  # evaluate mode
        if args.evaluate_coco_pretrained:
            if not args.eval_save_dir_custom:
                raise ValueError("eval_save_dir_custom arg must be provided "
                                 "to save metrics when evaluate_coco_pretrained is True") 
        else:
            if args.mode == "evaluate" and args.world_size != 1:
                raise ValueError("--nproc_per_node should be 1 for evaluate mode.")
            torch.backends.cudnn.deterministic = True
            if args.eval_checkpoint_path_custom:
                path_to_model_checkpoint = args.eval_checkpoint_path_custom
            else:
                path_to_model_checkpoint_folder = f"{args.save_log_folder}/exp{args.eval_exp_num}/model_checkpoints/"
                path_to_model_checkpoint = os.path.join(path_to_model_checkpoint_folder, f"{args.eval_checkpoint}.pth")
            checkpoint = torch.load(path_to_model_checkpoint,  map_location="cpu")
            state_dict = checkpoint['model']
            try:
                model_without_ddp.load_state_dict(state_dict)
            except:
                state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
                model_without_ddp.load_state_dict(state_dict)
            if args.save_onnx_during_eval:
                onnx_save_path = os.path.join(path_to_model_checkpoint_folder, f"{epoch}.onnx")
                utils.save_onnx_model(model.model, device, onnx_save_path)
        evaluate(model, data_loader_test, device=device, args=args)
        torch.cuda.empty_cache()

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Running mode
    parser.add_argument("--mode", dest="mode", type=str,
                        default="evaluate",
                        help="Modes: {train, evaluate, train_evaluate}")
    parser.add_argument("--eval-freq", dest="eval_freq", type=int,
                        default=2,
                        help="Frequency to evaluate with if mode = train_evaluate")

    # Training devicea
    parser.add_argument("--rank-0-gpu", dest="rank_0_gpu", type=int, default=0,
                        help="Which GPU to consider rank 0 for single-node, multi-worker, separate jobs."
                        "(rank 0 + world-size) gpu ids are used for the run.")
    parser.add_argument("--data-parallel", dest="data_parallel", action='store_true',
                        help="Whether to train in data parallel mode")
    parser.add_argument("--device-ids", dest="device_ids", type=list, default=[0],
                        help="gpu device id for training/evaluation")
    parser.add_argument("--num-workers", dest="num_workers", type=int, default=4,
                        help="Number of workers for data loading")
    parser.add_argument("--distributed", dest="distributed", type=bool, default=False,
                        help="Whether to perform data distributed training")
    parser.add_argument('--world-size', dest="world_size", default=1, type=int,
                        help='number of distributed processes')
    parser.add_argument('--dist-url', default='tcp://127.0.0.1:23456', help='url used to set up distributed training')

    # Experiment
    parser.add_argument("--exp-num", dest="exp_num", type=int, default=114,
                        help="exp id for tracking purposes")
    parser.add_argument("--resume", dest="resume", type=bool, default=False,
                        help="Whether to resume training an experiment")

    # Data
    # This is the folder where the training and validation data was
    # prepared and saved from prepare_clean_data.py
    rel_data_dir_train = "train_data_processed"
    rel_data_dir_test = "validation_data_processed"
    repo_root = os.path.dirname(os.path.abspath("../"))
    project_root = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--dataset", dest="dataset", type=str, default="arena",
                        help="arena or coco, will train and validate on respective train and val sets") 
    parser.add_argument("--project-root", dest="project_root", type=str, default=project_root,
                        help="Project root for running training and evaluation of vision models")
    parser.add_argument("--num-classes", dest="num_classes", type=int, default=62,
                        help="number of classes for the object detection model including background class")
    parser.add_argument("--data-root", dest="data_root", type=str, default=constants.DATA_ROOT,
                        help="path to image and metadata files root dir")
    parser.add_argument("--class-to-area-thresholds", dest="class_to_area_thresholds", type=str,
                        default=f"{repo_root}/data/vision-data/class_to_area_thresholds_customized.json",
                        help="Path to precomputed json file that maps classes to area thresholds for pruning")
    parser.add_argument("--train-metadata-file", dest="train_metadata_file", type=str,
                        default=f"{project_root}/{rel_data_dir_train}/metadata_train.txt",
                        help="path to text file with all metadata files")
    parser.add_argument("--test-metadata-file", dest="test_metadata_file", type=str,
                        default=f"{project_root}/{rel_data_dir_test}/metadata_test.txt",
                        help="path to text file with all metadata files")
    parser.add_argument("--class-to-idx-file", dest="class_to_idx_file", type=str,
                    default=f"{project_root}/{rel_data_dir_train}/class_to_idx.json",
                    help="Path to precomputed json file that maps classes to indices")
    parser.add_argument("--class-to-objectid-file", dest="class_to_objectid_file", type=str,
                    default=f"{project_root}/{rel_data_dir_train}/class_to_obj_id.json",
                    help="Path to precomputed json file that maps classes to object id")
    parser.add_argument("--objectid-to-class-file", dest="objectid_to_class_file", type=str,
                    default=f"{project_root}/{rel_data_dir_train}/obj_id_to_class.json",
                    help="Path to precomputed json file that maps object id to classes")
    parser.add_argument("--trajectory-data-class-file", dest="trajectory_data_class_file", type=str,
                    default=f"{project_root}/data/trajectory_data_classes.txt",
                    help="Path to list of classes to be included. If this is an empty list, all RG classes will be used.")
    # Coco dataset
    coco_data_root = "~/fiftyone/coco-2017/"
    parser.add_argument("--coco-data-root", dest="coco_data_root", type=str,
                        default=coco_data_root,
                        help="Data root for coco")
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

    # Model
    parser.add_argument("--visual-model", dest="visual_model", type=str, default="maskrcnn",
                        help="Type of model")
    parser.add_argument("--load-pretrained-model", dest="load_pretrained_model", type=bool, default=False,
                        help="Whether to start from moca/other custom pretrained model")
    parser.add_argument("--pretrained-model-path", dest="pretrained_model_path", type=str,
                        default="",
                        help="Path to pretrained model")
    parser.add_argument("--use-coco-pretrained-model", dest="use_coco_pretrained_model", type=bool,
                        default=False,
                        help="Whether to use coco pretrained model")
    parser.add_argument("--hidden-layer-size", dest="hidden_layer_size", type=int,
                        default=512,
                        help="Hidden layer size for mask predictor branch")
    # Hyperparameters
    parser.add_argument("--batch-size", dest="batch_size", type=int,
                        default=8,
                        help="Batch size for training and testing data loaders. For distributed mode, this is the batchsize per gpu")
    parser.add_argument("--lr", dest="lr", type=float,
                        default=0.00125,
                        help="Learning rate")
    parser.add_argument("--decay-epoch", dest="decay_epoch", type=int,
                        default=15,
                        help="learning rate is decayed every this many epochs")
    parser.add_argument("--decay-factor", dest="decay_factor", type=float,
                        default=0.1,
                        help="learning rate is decayed every decay_epochs by this factor")
    parser.add_argument("--momentum", dest="momentum", type=float,
                        default=0.9,
                        help="Momentum for optimizer")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float,
                        default=0.005,
                        help="Weight decay for optimizer")
    parser.add_argument("--num-epochs", dest="num_epochs", type=int,
                        default=55,
                        help="Number of epochs to train.")
    parser.add_argument("--optimizer", dest="optimizer", type=str,
                        default="SGD",
                        help="Optimization algo.")
    # Logs
    parser.add_argument("--save-log-folder", dest="save_log_folder", type=str,
                        default=constants.TRAINING_LOGS_ROOT,
                        help="Folder to save model checkpoints and training logs")
    # Evaluation
    parser.add_argument("--evaluate-coco-pretrained", dest="evaluate_coco_pretrained", type=bool,
                        default=False,
                        help="Whether to evaluate coco pretrained pytorch model")
    parser.add_argument("--eval-exp-num", dest="eval_exp_num", type=int,
                        default=112,
                        help="exp number to evaluate")
    parser.add_argument("--eval-checkpoint", dest="eval_checkpoint", type=int,
                        default=0,
                        help="checkpoint number of eval_exp_num to evaluate")
    parser.add_argument("--save-coco-metrics", dest="save_coco_metrics", type=bool,
                        default=True,
                        help="Whether to compute and save coco metrics")
    parser.add_argument("--save-images-masks-to-disk", dest="save_images_masks_to_disk", type=bool,
                        default=False,
                        help="Whether to save images and predicted and gt masks to disk during evaluation")
    parser.add_argument("--save-onnx-during-eval", dest="save_onnx_during_eval", action='store_true',
                        help="Whether to save onnx model during evaluation")
    parser.add_argument("--eval-checkpoint-path-custom", dest="eval_checkpoint_path_custom", type=str,
                        default="",
                        help="If absolute custom checkpoint path is given (not in training logs)")
    parser.add_argument("--eval-save-dir-custom", dest="eval_save_dir_custom", type=str,
                        default="",
                        help="Custom folder to save all evaluation metrics.") 

    args = parser.parse_args()
    main(args)
