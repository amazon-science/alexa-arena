# Copyright Amazon.com, Inc. or its affiliates. All Rights Reserved.
# SPDX-License-Identifier: LGPL-2.1

import json
import os
from argparse import ArgumentParser
from tabnanny import check

import numpy as np
import torch
import torch.nn as nn
import torch.cuda.amp as amp
import utils
from functools import partial
from data_generators.vl_data_generator import ArenaRefDataset
from engine import train, validate
from torch.optim.lr_scheduler import MultiStepLR
import shutil
import resource
from models import build_predictor
from utils import worker_init_fn
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8000, rlimit[1]))


def main(args):
    utils.init_distributed_mode(args)
    if args.mode in {"evaluate", "train_evaluate"}:
        raise NotImplementedError("Evaluation mode not implemented. "
                                 "Please run end-to-end mission level evaluation after training.")
    training = True if (args.mode in {"train", "train_evaluate"}) else False
    # use our dataset and defined transformations
    if training:
        dataset = ArenaRefDataset(
            metadata_file=args.train_metadata_file,
            images_root=args.images_root,
            split="train",
            args=args)
        init_fn = partial(worker_init_fn,
                          num_workers=args.num_workers,
                          rank=args.rank,
                          seed=0)
        train_sampler = torch.utils.data.distributed.DistributedSampler(dataset, shuffle=True)
        data_loader = torch.utils.data.DataLoader(
            dataset,
            sampler=train_sampler,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=init_fn,
            collate_fn=utils.collate_fn,
            drop_last=True,
            pin_memory=True
            )
    if args.mode in {"evaluate", "train_evaluate"}:
        dataset_test = ArenaRefDataset(
            metadata_file=args.test_metadata_file,
            images_root=args.images_root,
            split="validation",
            args=args)
        test_sampler = torch.utils.data.distributed.DistributedSampler(dataset_test, shuffle=False)
        data_loader_test = torch.utils.data.DataLoader(
            dataset_test,
            batch_size=16,
            shuffle=False,
            sampler=test_sampler,
            num_workers=args.num_workers,
            drop_last=False,
            pin_memory=True)

    # get the model using our helper function
    model, param_list = build_predictor(args)
    if args.sync_bn:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    if args.distributed:
        model = nn.parallel.DistributedDataParallel(model.cuda(),
                                                    device_ids=[args.gpu],
                                                    find_unused_parameters=True)
        model_without_ddp = model.module
    print("Model loaded!")
    device = "cuda"
    model.to(device)
    params = filter(lambda p: p.requires_grad, model.parameters())
    num_params = sum([np.prod(p.size()) for p in params])
    print("Number of model params: ", num_params)
    optimizer = torch.optim.Adam(param_list,
                                 lr=args.base_lr,
                                 weight_decay=args.weight_decay)
    scheduler = MultiStepLR(optimizer,
                           milestones=args.milestones,
                           gamma=args.lr_decay)
    scaler = amp.GradScaler()
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
    best_iou = 0.0
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
            best_iou = checkpoint['best_iou']
            try:
                model_without_ddp.load_state_dict(state_dict)            
            except:
                state_dict = {k.partition('module.')[2]: state_dict[k] for k in state_dict.keys()}
                model_without_ddp.load_state_dict(state_dict)
            optimizer.load_state_dict(checkpoint["optim"])
            scheduler.load_state_dict(checkpoint['scheduler'])

        print(f"Launching training experiment {args.exp_num}!")
        for epoch in range(start_epoch, num_epochs):
            train_sampler.set_epoch(epoch + 1)
            optimizer.zero_grad()
            print(f"\nEpoch {epoch}")
            # train for one epoch, printing every print-freq iterations
            loss_meter, iou_meter, pr_meter = train(data_loader, model, optimizer, scheduler, scaler, epoch + 1,
                  args)
            iou = 0
            prec_dict = {}
            if (args.mode == "train_evaluate") and ((epoch + 1) % args.eval_freq == 0):
                iou, prec_dict = validate(data_loader_test, model, epoch + 1, args)

            if utils.is_main_process():
                fsave = os.path.join(exp_save_log_folder, "model_checkpoints", '{}.pth'.format(epoch))
                torch.save({
                    'iou': iou,
                    'best_iou': best_iou,
                    'prec': prec_dict,
                    'model': model_without_ddp.state_dict(),
                    'optim': optimizer.state_dict(),
                    'scheduler': scheduler.state_dict(),
                    'epoch': epoch,
                    'args': args
                    }, fsave)
                if iou >= best_iou:
                    best_iou = iou
                fbest = os.path.join(stats_save_path, f"{epoch}.json")
                stats = {
                    'running_loss_avg': loss_meter.avg,
                    'running_iou_avg': iou_meter.avg,
                    'running_prec_at_50_avg': pr_meter.avg,
                }
                with open(fbest, 'wt') as f:
                    json.dump(stats, f, indent=2)
            scheduler.step()
            torch.cuda.empty_cache()
        print("Training complete.")

if __name__ == "__main__":
    parser = ArgumentParser()
    
    # Running mode
    parser.add_argument("--mode", dest="mode", type=str,
                        default="train",
                        help="Modes: {train, evaluate, train_evaluate}")
    parser.add_argument("--eval-freq", dest="eval_freq", type=int,
                        default=10,
                        help="Frequency to evaluate with if mode = train_evaluate")
    parser.add_argument("--print-freq", dest="print_freq", type=int,
                        default=10,
                        help="Frequency to print logs in steps")

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
    repo_root = os.path.dirname(os.path.abspath("../"))
    data_root = "{}/data/trajectory-data/".format(repo_root)
    project_root = os.path.dirname(os.path.abspath(__file__))
    parser.add_argument("--project-root", dest="project_root", type=str, default=project_root,
                        help="Project root for running training and evaluation of vision models")
    parser.add_argument("--images-root", dest="images_root", type=str,
                        default=f"{data_root}/mission_images",
                        help="path to image and metadata files root dir")
    parser.add_argument("--train-metadata-file", dest="train_metadata_file", type=str,
                        default=f"{data_root}/train.json",
                        help="path to text file with all metadata files")
    parser.add_argument("--test-metadata-file", dest="test_metadata_file", type=str,
                        default=f"{data_root}/valid.json",
                        help="path to text file with all metadata files")
    parser.add_argument("--class-to-idx-file", dest="class_to_idx_file", type=str,
                        default=f"{project_root}/data_generators/resources/class_to_idx.json",
                        help="Path to precomputed json file that maps classes to indices")
    parser.add_argument("--obj-id-to-class-file", dest="obj_id_to_class_file", type=str,
                        default=f"{project_root}/data_generators/resources/obj_id_to_class_customized.json",
                        help="Path to precomputed json file that maps classes to object id")

    # Model
    parser.add_argument("--clip-pretrain", dest="clip_pretrain", type=str,
                        default=os.path.join(project_root, "pretrained/RN50.pt"),
                        help="Path to pretrained clip model")
    parser.add_argument("--word-len", dest="word_len", type=int, default=22,
                        help="Maximum number of words in language prompt")
    parser.add_argument("--word-dim", dest="word_dim", type=int, default=1024,
                        help="Word dimension")
    parser.add_argument("--fpn-in", dest="fpn_in", type=list, default= [512, 1024, 1024],
                        help="Fpn in")
    parser.add_argument("--fpn-out", dest="fpn_out", type=list, default=[256, 512, 1024],
                        help="Fpn out")
    parser.add_argument("--num-layers", dest="num_layers", type=int, default=3,
                        help="Number of layers")
    parser.add_argument("--vis-dim", dest="vis_dim", type=int, default=512,
                        help="Visual feature dims")
    parser.add_argument("--num-head", dest="num_head", type=int, default=8,
                        help="Number of heads for trans")
    parser.add_argument("--dim-ffn", dest="dim_ffn", type=int, default=2048,
                        help="Feed forward network dim")
    parser.add_argument("--dropout", dest="dropout", type=float, default=0.1,
                        help="Dropout prob")
    parser.add_argument("--intermediate", dest="intermediate", type=bool, default=False,
                        help="Intermediate")
    parser.add_argument("--sync-bn", dest="sync_bn", type=bool, default=True,
                        help="Batchnormalization sync between processes")

    parser.add_argument("--hidden-layer-size", dest="hidden_layer_size", type=int,
                        default=512,
                        help="Hidden layer size for mask predictor branch")
    # Hyperparameters
    parser.add_argument("--batch-size", dest="batch_size", type=int,
                        default=8,
                        help="Batch size for training and testing data loaders. "
                             "For distributed mode, this is the batchsize per gpu")
    parser.add_argument("--lr", dest="lr", type=float,
                        default=0.00125,
                        help="Learning rate")
    parser.add_argument("--lr-multi", dest="lr_multi", type=float,
                        default=0.1,
                        help="Learning rate multiplier for clip backbone")
    parser.add_argument("--lr-seq", dest="lr_seq", type=float,
                        default=0.0001,
                        help="Learning rate for sequential act obj pred")
    parser.add_argument("--base-lr", dest="base_lr", type=float,
                        default=0.00001,
                        help="Learning rate head")
    parser.add_argument("--decay-epoch", dest="decay_epoch", type=int,
                        default=25,
                        help="learning rate is decayed every this many epochs")
    parser.add_argument("--lr-decay", dest="lr_decay", type=float,
                        default=0.1,
                        help="learning rate is decayed every decay_epochs by this factor")
    parser.add_argument("--milestones", dest="milestones", type=list,
                        default=[35],
                        help="Learning rate schedule milestones to decay lr")
    parser.add_argument("--momentum", dest="momentum", type=float,
                        default=0.9,
                        help="Momentum for optimizer")
    parser.add_argument("--weight-decay", dest="weight_decay", type=float,
                        default=0.00001,
                        help="Weight decay for optimizer")
    parser.add_argument("--num-epochs", dest="num_epochs", type=int,
                        default=75,
                        help="Number of epochs to train.")
    parser.add_argument("--optimizer", dest="optimizer", type=str,
                        default="SGD",
                        help="Optimization algo.")
    parser.add_argument("--max-norm", dest="max_norm", type=float,
                        default=0.0,
                        help="Grad clip norm")
    # Logs
    parser.add_argument("--save-log-folder", dest="save_log_folder", type=str,
                        default="/home/ubuntu/training_logs/",
                        help="Folder to save model checkpoints and training logs")

    args = parser.parse_args()
    main(args)
