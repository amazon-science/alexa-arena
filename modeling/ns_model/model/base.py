# Modified from https://github.com/alexpashevich/E.T./blob/master/alfred/model/base.py - Licensed under the
# MIT License.
#
# Modifications Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.


from torch import nn
from modeling.ns_model.utils import data_util

class Model(nn.Module):
    def __init__(self, args, embs_ann, vocab_out, pad, seg):
        '''
        Abstract model
        '''
        nn.Module.__init__(self)
        self.args = args
        self.vocab_out = vocab_out
        self.pad, self.seg = pad, seg
        self.visual_tensor_shape = data_util.get_feat_shape(self.args.visual_archi)[1:]

        # create language and action embeddings
        self.embs_ann = nn.ModuleDict({})
        for emb_name, emb_size in embs_ann.items():
            self.embs_ann[emb_name] = nn.Embedding(emb_size, args.demb)

        # dropouts
        self.dropout_vis = nn.Dropout(args.dropout['vis'], inplace=True)
        self.dropout_lang = nn.Dropout2d(args.dropout['lang'])

    def init_weights(self, init_range=0.1):
        '''
        init linear layers in embeddings
        '''
        for emb_ann in self.embs_ann.values():
            emb_ann.weight.data.uniform_(-init_range, init_range)

    def compute_metrics(self, model_out, gt_dict, metrics_dict, verbose):
        '''
        compute model-specific metrics and put it to metrics dict
        '''
        raise NotImplementedError

    def forward(self, vocab, **inputs):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        raise NotImplementedError()

    def compute_batch_loss(self, model_out, gt_dict):
        '''
        compute the loss function for a single batch
        '''
        raise NotImplementedError()

    def compute_loss(self, model_outs, gt_dicts):
        '''
        compute the loss function for several batches
        '''
        # compute losses for each batch
        losses = {}
        for dataset_key in model_outs.keys():
            losses[dataset_key] = self.compute_batch_loss(
                model_outs[dataset_key], gt_dicts[dataset_key])
        return losses
