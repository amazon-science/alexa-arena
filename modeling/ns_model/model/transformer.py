# Modified from https://github.com/alexpashevich/E.T./blob/master/alfred/model/transformer.py - Licensed under the
# MIT License.
#
# Modifications Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.

import torch
from torch import nn
from torch.nn import functional as F

from modeling.ns_model.model import base
from modeling.ns_model.nn.enc_lang import EncoderLang
from modeling.ns_model.nn.enc_visual import FeatureFlat
from modeling.ns_model.nn.enc_vl import EncoderVL
from modeling.ns_model.nn.encodings import DatasetLearnedEncoding
from modeling.ns_model.nn.dec_object import ObjectClassifier
from modeling.ns_model.utils import model_util


class Model(base.Model):
    def __init__(self, args, embs_ann, vocab_out, pad, seg):
        '''
        transformer agent
        '''
        super().__init__(args, embs_ann, vocab_out, pad, seg)

        # encoder and visual embeddings
        self.encoder_vl = EncoderVL(args)
        # pre-encoder for language tokens
        self.encoder_lang = EncoderLang(
            args.encoder_lang['layers'], args, embs_ann)
        # feature embeddings
        self.vis_feat = FeatureFlat(
            input_shape=self.visual_tensor_shape,
            output_size=args.demb)
        # dataset id learned encoding (applied after the encoder_lang)
        self.dataset_enc = None
        if args.enc['dataset']:
            self.dataset_enc = DatasetLearnedEncoding(args.demb, args.data['train'])
        # embeddings for actions
        self.emb_action = nn.Embedding(len(vocab_out), args.demb)
        # dropouts
        self.dropout_action = nn.Dropout1d(args.dropout['transformer']['action'])

        # decoder parts
        encoder_output_size = args.demb
        self.dec_action = nn.Linear(
            encoder_output_size, args.demb)
        self.dec_object = ObjectClassifier(encoder_output_size)

        # skip connection for object predictions
        self.object_feat = FeatureFlat(
            input_shape=self.visual_tensor_shape,
            output_size=args.demb)

        # final touch
        self.init_weights()
        self.reset()

    def forward(self, vocab, **inputs):
        '''
        forward the model for multiple time-steps (used for training)
        '''
        # embed language
        output = {}
        emb_lang, lengths_lang = self.embed_lang(inputs['lang'], vocab)
        emb_lang = self.dataset_enc(emb_lang, vocab) if self.dataset_enc else emb_lang

        # embed frames and actions
        emb_frames, emb_object = self.embed_frames(inputs['frames'])
        lengths_frames = inputs['lengths_frames']
        emb_actions = self.embed_actions(inputs['action'])

        assert emb_frames.shape == emb_actions.shape
        lengths_actions = lengths_frames.clone()
        length_frames_max = inputs['length_frames_max']

        # concatenate language, frames and actions and add encodings
        encoder_out, _ = self.encoder_vl(
            emb_lang, emb_frames, emb_actions, lengths_lang,
            lengths_frames, lengths_actions, length_frames_max)
        # use outputs corresponding to visual frames for prediction only
        encoder_out_visual = encoder_out[
            :, lengths_lang.max().item():
            lengths_lang.max().item() + length_frames_max]

        # get the output actions
        decoder_input = encoder_out_visual.reshape(-1, self.args.demb)
        action_emb_flat = self.dec_action(decoder_input)
        action_flat = action_emb_flat.mm(self.emb_action.weight.t())
        action = action_flat.view(
            *encoder_out_visual.shape[:2], *action_flat.shape[1:])

        # get the output objects
        emb_object_flat = emb_object.view(-1, self.args.demb)
        decoder_input = decoder_input + emb_object_flat
        object_flat = self.dec_object(decoder_input)
        objects = object_flat.view(
            *encoder_out_visual.shape[:2], *object_flat.shape[1:])
        output.update({'action': action, 'object': objects})

        return output

    def embed_lang(self, lang_pad, vocab):
        '''
        take a list of annotation tokens and extract embeddings with EncoderLang
        '''
        # assert lang_pad.max().item() < len(vocab) 
        embedder_lang = self.embs_ann[vocab.name]
        emb_lang, lengths_lang = self.encoder_lang(
            lang_pad, embedder_lang, vocab, self.pad)
        if self.args.detach_lang_emb:
            emb_lang = emb_lang.clone().detach()
        return emb_lang, lengths_lang

    def embed_frames(self, frames_pad):
        '''
        take a list of frames tensors, pad it, apply dropout and extract embeddings
        '''
        self.dropout_vis(frames_pad)
        frames_4d = frames_pad.view(-1, *frames_pad.shape[2:])
        frames_pad_emb = self.vis_feat(frames_4d).view(
            *frames_pad.shape[:2], -1)
        frames_pad_emb_skip = self.object_feat(
            frames_4d).view(*frames_pad.shape[:2], -1)
        return frames_pad_emb, frames_pad_emb_skip

    def embed_actions(self, actions):
        '''
        embed previous actions
        '''
        emb_actions = self.emb_action(actions)
        emb_actions = self.dropout_action(emb_actions)
        return emb_actions

    def reset(self):
        '''
        reset internal states (used for real-time execution during eval)
        '''
        self.frames_traj = torch.zeros(1, 0, *self.visual_tensor_shape)
        self.action_traj = torch.zeros(1, 0).long()

    def step(self, input_dict, vocab, prev_action=None):
        '''
        forward the model for a single time-step (used for real-time execution during eval)
        '''
        frames = input_dict['frames']
        device = frames.device
        if prev_action is not None:
            prev_action_int = vocab['action_low'].word2index[prev_action]
            prev_action_tensor = torch.tensor(prev_action_int)[None, None].to(device)
            self.action_traj = torch.cat(
                (self.action_traj.to(device), prev_action_tensor), dim=1)
        self.frames_traj = torch.cat(
            (self.frames_traj.to(device), frames), dim=1)
        # at timestep t we have t-1 prev actions so we should pad them
        action_traj_pad = torch.cat(
            (self.action_traj.to(device),
             torch.zeros((1, 1)).to(device).long()), dim=1)
        model_out = self.forward(
            vocab=vocab['word'],
            lang=input_dict['lang'],
            lengths_lang=input_dict['lengths_lang'],
            length_lang_max=input_dict['length_lang_max'],
            frames=self.frames_traj.clone(),
            lengths_frames=torch.tensor([self.frames_traj.size(1)]),
            length_frames_max=self.frames_traj.size(1),
            action=action_traj_pad)
        step_out = {}
        for key, value in model_out.items():
            # return only the last actions, ignore the rest
            step_out[key] = value[:, -1:]
        return step_out

    def compute_batch_loss(self, model_out, gt_dict):
        '''
        loss function for Seq2Seq agent
        '''
        losses = dict()

        # action loss
        action_pred = model_out['action'].view(-1, model_out['action'].shape[-1])
        action_gt = gt_dict['action'].view(-1)
        pad_mask = (action_gt != self.pad)
        action_loss = F.cross_entropy(action_pred, action_gt, reduction='none')
        action_loss *= pad_mask.float()
        action_loss = action_loss.mean()
        losses['action'] = action_loss * self.args.action_loss_wt

        # object classes loss
        object_pred = model_out['object']
        object_gt = torch.cat(gt_dict['object'], dim=0)
        interact_idxs = gt_dict['action_valid_interact'].view(-1).nonzero(
            as_tuple=False).view(-1)
        if interact_idxs.nelement() > 0:
            object_pred = object_pred.view(
                object_pred.shape[0] * object_pred.shape[1],
                *object_pred.shape[2:])
            object_loss = model_util.obj_classes_loss(
                object_pred, object_gt, interact_idxs)
            losses['object'] = object_loss * self.args.object_loss_wt

        return losses

    def init_weights(self, init_range=0.1):
        '''
        init embeddings uniformly
        '''
        super().init_weights(init_range)
        self.dec_action.bias.data.zero_()
        self.dec_action.weight.data.uniform_(-init_range, init_range)
        self.emb_action.weight.data.uniform_(-init_range, init_range)

    def compute_metrics(self, model_out, gt_dict, metrics_dict, verbose=False):
        '''
        compute exact matching and f1 score for action predictions
        '''
        preds = model_util.extract_action_preds(
            model_out, self.pad, self.vocab_out, lang_only=True)
        stop_token = self.vocab_out.word2index['<<stop>>']
        gt_actions = model_util.tokens_to_lang(
            gt_dict['action'], self.vocab_out, {self.pad, stop_token})
        model_util.compute_f1_and_exact(
            metrics_dict, [p['action'] for p in preds], gt_actions, 'action')
        model_util.compute_obj_class_precision(
            metrics_dict, gt_dict, model_out['object'])
