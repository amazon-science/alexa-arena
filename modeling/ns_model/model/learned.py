# Modified from https://github.com/alexpashevich/E.T./blob/master/alfred/model/learned.py - Licensed under the
# MIT License.
#
# Modifications Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.


import os
import torch
import json
import collections
import gtimer as gt

from tqdm import tqdm
from importlib import import_module
from torch import nn

from modeling.ns_model.utils import data_util, model_util


class LearnedModel(nn.Module):
    def __init__(self, args, embs_ann, vocab_in, vocab_out):
        '''
        Abstract model
        '''
        nn.Module.__init__(self)
        self.args = args
        self.embs_ann = embs_ann
        self.vocab_in = vocab_in
        self.vocab_out = vocab_out
        # sentinel tokens
        self.pad, self.seg = 0, 1
        # create the model to be trained
        ModelClass = import_module('modeling.ns_model.model.{}'.format(args.model)).Model
        self.model = ModelClass(args, embs_ann, vocab_out, self.pad, self.seg)

    def run_train(self, loaders, info, optimizer=None):
        '''
        training loop
        '''
        # prepare dictionaries
        loaders_train = dict(filter(lambda x: 'train' in x[0], loaders.items()))
        assert len(set([len(loader) for loader in loaders_train.values()])) == 1
        loaders_valid = dict(filter(lambda x: 'train' not in x[0], loaders.items()))
        vocabs_in = {'{};{}'.format(
            loader.dataset.name, loader.dataset.ann_type): loader.dataset.vocab_in
                     for loader in loaders.values()}
        epoch_length = len(next(iter(loaders_train.values())))
        # dump config
        with open(os.path.join(self.args.dout, 'config.json'), 'wt') as f:
            json.dump(vars(self.args), f, indent=2)
        # optimizer
        optimizer, schedulers = model_util.create_optimizer_and_schedulers(
            info['progress'], self.args, self.parameters(), optimizer)
        # make sure that all train loaders have the same length
        assert len(set([len(loader) for loader in loaders_train.values()])) == 1
        model_util.save_log(
            self.args.dout, progress=info['progress'], total=self.args.epochs,
            stage='train', best_loss=info['best_loss'], iters=info['iters'])
        
        all_stats = []

        # display dout
        print("Saving to: %s" % self.args.dout)
        for epoch in range(info['progress'], self.args.epochs):
            print('Epoch {}/{}'.format(epoch, self.args.epochs))
            self.train()
            train_iterators = {
                key: iter(loader) for key, loader in loaders_train.items()}
            metrics = {key: collections.defaultdict(list) for key in loaders_train}
            gt.reset()

            for _ in tqdm(range(epoch_length), desc='train'):
                # sample batches
                batches = data_util.sample_batches(
                    train_iterators, self.vocab_in, self.vocab_out, self.args.device, self.pad, self.args)
                gt.stamp('data fetching', unique=False)
                
                # do the forward passes
                model_outs, losses_train = {}, {}
                for batch_name, (input_dict, gt_dict) in batches.items():
                    model_outs[batch_name] = self.model.forward(
                        self.vocab_in,
                        action=gt_dict['action'], **input_dict)
                gt.stamp('forward pass', unique=False)
                # compute losses
                losses_train = self.model.compute_loss(
                    model_outs,
                    {key: gt_dict for key, (_, gt_dict) in batches.items()})

                # do the gradient step
                optimizer.zero_grad()
                sum_loss = sum(
                    [sum(loss.values()) for name, loss in losses_train.items()])
                sum_loss.backward()
                optimizer.step()
                gt.stamp('optimizer', unique=False)

                # compute metrics
                for dataset_name in losses_train.keys():
                    self.model.compute_metrics(
                        model_outs[dataset_name], batches[dataset_name][1],
                        metrics[dataset_name])
                    for key, value in losses_train[dataset_name].items():
                        metrics[dataset_name]['loss/' + key].append(
                            value.item())
                    metrics[dataset_name]['loss/total'].append(
                        sum_loss.detach().cpu().item())
                gt.stamp('metrics', unique=False)

            # compute metrics for train
            print('Computing train and validation metrics...')
            metrics = {data: {k: sum(v) / len(v) for k, v in metr.items()}
                       for data, metr in metrics.items()}

            stats = {'epoch': epoch, 'general': {
                'learning_rate': optimizer.param_groups[0]['lr']}, **metrics}

            # save the checkpoint
            print('Saving models...')
            model_util.save_model(
                self, 'model_{:02d}.pth'.format(epoch), stats, optimizer=optimizer)
            model_util.save_model(self, 'latest.pth', stats, symlink=True)

            all_stats.append(stats)
            # write averaged stats
            with open(self.args.checkpt_dir + str(self.args.exp_num) + "/stats.json", "w") as f:
                json.dump(all_stats, f, indent=4)

            # dump the training info
            model_util.save_log(
                self.args.dout, progress=epoch+1, total=self.args.epochs,
                stage='train', best_loss=info['best_loss'], iters=info['iters'])

            model_util.adjust_lr(optimizer, self.args, epoch, schedulers)

        print('{} epochs are completed, all the models were saved to: {}'.format(
            self.args.epochs, self.args.dout))

