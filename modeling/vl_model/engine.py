# Modified from https://github.com/DerrickWang005/CRIS.pytorch/blob/master/engine/engine.py - Licensed under the
# MIT License.
#
# Modifications Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.


import time
import torch
import torch.cuda.amp as amp
import torch.distributed as dist
import torch.nn.functional as F
import numpy as np
from tqdm import tqdm
import utils
from data_generators.vl_data_generator import tokenize
from utils import (AverageMeter, ProgressMeter, concat_all_gather,
                   trainMetricGPU, action_object_eos_accuracy)
from loguru import logger
import resource
import cv2
rlimit = resource.getrlimit(resource.RLIMIT_NOFILE)
resource.setrlimit(resource.RLIMIT_NOFILE, (8000, rlimit[1]))


def train(train_loader, model, optimizer, scheduler, scaler, epoch, args):
    batch_time = AverageMeter('Batch', ':2.2f')
    data_time = AverageMeter('Data', ':2.2f')
    lr = AverageMeter('Lr', ':1.6f')
    loss_meter = AverageMeter('Loss', ':2.4f')
    mask_loss_meter = AverageMeter('Mask Loss', ':2.4f')
    action_loss_meter = AverageMeter('Action Loss', ':2.4f')
    object_loss_meter = AverageMeter('Object Loss', ':2.4f')
    eos_loss_meter = AverageMeter('EOS Loss', ':2.4f')
    iou_meter = AverageMeter('IoU', ':2.2f')
    pr_meter = AverageMeter('Prec@50', ':2.2f')
    action_object_eos_accuracy_meter = AverageMeter('ActionObjEosAcc', ':2.2f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, lr, loss_meter, mask_loss_meter, action_loss_meter, object_loss_meter, eos_loss_meter,
        iou_meter, pr_meter, action_object_eos_accuracy_meter],
        prefix="Training: Epoch=[{}/{}] ".format(epoch, args.num_epochs))

    model.train()
    time.sleep(2)
    end = time.time()

    for i, (text, sequential_data) in enumerate(train_loader):
        data_time.update(time.time() - end)
        # data
        images, texts, targets, actions_objects = [], [], [], []
        for d_num, data in enumerate(sequential_data):
            image = torch.cat([img.unsqueeze(0) for img in data['images']], dim=0)
            target = torch.cat([msk.unsqueeze(0) for msk in data['masks']], dim=0)
            text_inp = text[d_num].unsqueeze(0).repeat(image.shape[0], 1)
            action_object_pairs = torch.cat([torch.tensor(elt).unsqueeze(0) for elt in data['action_object_pairs']])

            image = image.cuda(non_blocking=True)
            text_inp = text_inp.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True).unsqueeze(1)
            action_object_pairs = action_object_pairs.cuda(non_blocking=True)
            images.append(image)
            texts.append(text_inp)
            targets.append(target)
            actions_objects.append(action_object_pairs)
        # forward
        with amp.autocast():
            images = torch.cat(images)
            texts = torch.cat(texts)
            targets = torch.cat(targets)
            pred, target, mask_loss, action_loss, object_loss, eos_loss, loss, action_logits, object_logits, eos_logits = \
                model(img=images, word=texts, mask=targets, action_object_pairs=actions_objects)

        # backward
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        if args.max_norm:
            torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_norm)
        scaler.step(optimizer)
        scaler.update()

        # metric
        accuracy = action_object_eos_accuracy(action_logits, object_logits, eos_logits, actions_objects)
        iou, pr5 = trainMetricGPU(pred, target, 0.35, 0.5)
        dist.all_reduce(loss.detach())
        dist.all_reduce(mask_loss.detach())
        dist.all_reduce(action_loss.detach())
        dist.all_reduce(object_loss.detach())
        dist.all_reduce(eos_loss.detach())
        dist.all_reduce(iou)
        dist.all_reduce(pr5)
        dist.all_reduce(accuracy)
        loss = loss / dist.get_world_size()
        iou = iou / dist.get_world_size()
        pr5 = pr5 / dist.get_world_size()
        accuracy = accuracy / dist.get_world_size()

        loss_meter.update(loss.item(), image.size(0))
        mask_loss_meter.update(mask_loss.item(), image.size(0))
        action_loss_meter.update(action_loss.item(), image.size(0))
        object_loss_meter.update(object_loss.item(), image.size(0))
        eos_loss_meter.update(eos_loss.item(), image.size(0))
        iou_meter.update(iou.item(), image.size(0))
        pr_meter.update(pr5.item(), image.size(0))
        action_object_eos_accuracy_meter.update(accuracy.item(), len(sequential_data))
        lr.update(scheduler.get_last_lr()[-1])
        batch_time.update(time.time() - end)
        end = time.time()

        if (i + 1) % args.print_freq == 0 and utils.is_main_process():
            progress.display(i + 1)
    meters = [loss_meter, iou_meter, pr_meter]
    return meters

@torch.no_grad()
def validate(val_loader, model, epoch, args):
    iou_list = []
    model.eval()
    time.sleep(2)
    for imgs, texts, param in val_loader:
        # data
        imgs = imgs.cuda(non_blocking=True)
        texts = texts.cuda(non_blocking=True)
        # inference
        preds = model(imgs, texts)
        preds = torch.sigmoid(preds)
        if preds.shape[-2:] != imgs.shape[-2:]:
            preds = F.interpolate(preds,
                                  size=imgs.shape[-2:],
                                  mode='bicubic',
                                  align_corners=True).squeeze(1)
        # process one batch
        for pred, mask_dir, mat, ori_size, gt_mask in zip(preds, param['mask_dir'],
                                                 param['inverse'],
                                                 param['ori_size'],
                                                 param['mask']):
            h, w = np.array(ori_size)
            mat = np.array(mat)
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            pred = np.array(pred > 0.35)
            # iou
            mask = gt_mask.numpy()
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(imgs.device)
    iou_list = concat_all_gather(iou_list)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    temp = '  '
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres * 10)
        value = prec_list[i].item()
        prec[key] = value
        temp += "{}: {:.2f}  ".format(key, 100. * value)
    head = 'Evaluation: Epoch=[{}/{}]  IoU={:.2f}'.format(
        epoch, args.num_epochs, 100. * iou.item())
    logger.info(head + temp)
    return iou.item(), prec

@torch.no_grad()
def inference(test_loader, model, args):
    iou_list = []
    tbar = tqdm(test_loader, desc='Inference:', ncols=100)
    model.eval()
    time.sleep(2)
    for img, param in tbar:
        # data
        img = img.cuda(non_blocking=True)
        mask = param['mask']
        # multiple sentences
        for sent in param['sents']:
            text = tokenize(sent, args.word_len, True)
            text = text.cuda(non_blocking=True)
            # inference
            pred = model(img, text)
            pred = torch.sigmoid(pred)
            if pred.shape[-2:] != img.shape[-2:]:
                pred = F.interpolate(pred,
                                     size=img.shape[-2:],
                                     mode='bicubic',
                                     align_corners=True).squeeze()
            # process one sentence
            h, w = param['ori_size'].numpy()[0]
            mat = param['inverse'].numpy()[0]
            pred = pred.cpu().numpy()
            pred = cv2.warpAffine(pred, mat, (w, h),
                                  flags=cv2.INTER_CUBIC,
                                  borderValue=0.)
            mask = param['mask'][0].numpy()
            pred = np.array(pred > 0.35)
            # iou
            inter = np.logical_and(pred, mask)
            union = np.logical_or(pred, mask)
            iou = np.sum(inter) / (np.sum(union) + 1e-6)
            iou_list.append(iou)
    logger.info('=> Metric Calculation <=')
    iou_list = np.stack(iou_list)
    iou_list = torch.from_numpy(iou_list).to(img.device)
    prec_list = []
    for thres in torch.arange(0.5, 1.0, 0.1):
        tmp = (iou_list > thres).float().mean()
        prec_list.append(tmp)
    iou = iou_list.mean()
    prec = {}
    for i, thres in enumerate(range(5, 10)):
        key = 'Pr@{}'.format(thres*10)
        value = prec_list[i].item()
        prec[key] = value
    logger.info('IoU={:.2f}'.format(100.*iou.item()))
    for k, v in prec.items():
        logger.info('{}: {:.2f}.'.format(k, 100.*v))

    return iou.item(), prec
