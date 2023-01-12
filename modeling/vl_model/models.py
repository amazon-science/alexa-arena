# Modified from https://github.com/DerrickWang005/CRIS.pytorch/blob/master/model/clip.py 
# and https://github.com/DerrickWang005/CRIS.pytorch/blob/master/model/segmenter.py 
# Licensed under the MIT License.
#
# Modifications Copyright 2023 Amazon.com, Inc. or its affiliates. All Rights Reserved.


import os
import sys
from typing import Tuple, Union

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from clip.model import (CLIP, AttentionPool2d, ModifiedResNet, Transformer,
                        convert_weights)
from loguru import logger

sys.path.append(f"{os.environ['HOME']}/AlexaArena/")
import modeling.vl_model.utils as utils
from modeling.vl_model.layers import FPN, Projector, TransformerDecoder


def build_predictor(args):
    model = ActionMaskPredictionModel(args)
    backbone = []
    head = []
    action_object_predictors = []
    for k, v in model.named_parameters():
        if k.startswith('backbone') and 'positional_embedding' not in k:
            backbone.append(v)
        elif ("action_object_predictor" in k) or ("action_classifier" in k) or ("object_classifier" in k) or \
            ("eos_classifier" in k):
            action_object_predictors.append(v)
        else:
            head.append(v)
    if utils.is_main_process():
        logger.info('Backbone with decay={}, Head={}'.format(len(backbone), len(head)))
    param_list = [{
        'params': backbone,
        'initial_lr': args.lr_multi * args.base_lr
    }, {
        'params': head,
        'initial_lr': args.base_lr
    }, {
        'params': action_object_predictors,
        'initial_lr': args.lr_seq
    }]
    return model, param_list


class ActionMaskPredictionModel(nn.Module):
    # Modified from https://github.com/DerrickWang005/CRIS.pytorch/blob/master/model/segmenter.py
    def __init__(self, cfg):
        super().__init__()
        # Vision & Text Encoder
        clip_model = torch.jit.load(cfg.clip_pretrain,
                                    map_location="cpu").eval()
        self.backbone = build_model(clip_model.state_dict(), cfg.word_len).float()
        # Multi-Modal FPN
        self.neck = FPN(in_channels=cfg.fpn_in, out_channels=cfg.fpn_out)
        # Decoder
        self.decoder = TransformerDecoder(num_layers=cfg.num_layers,
                                          d_model=cfg.vis_dim,
                                          nhead=cfg.num_head,
                                          dim_ffn=cfg.dim_ffn,
                                          dropout=cfg.dropout,
                                          return_intermediate=cfg.intermediate)
        # Projector
        self.proj = Projector(cfg.word_dim + 512, cfg.vis_dim // 2, 3)
        # Given text, img vector input, previous action and object pair, predict current action and object pair
        input_size, hidden_size, num_layers = 512, 512, 2
        self.action_object_predictor = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True, dropout=0.1)
        num_actions, num_objects = 12, 94 # also includes rooms and turn right, left
        self.action_classifier = nn.Linear(512, num_actions)
        self.object_classifier =  nn.Linear(512, num_objects)
        self.eos_classifier = nn.Linear(512, 1)
        self.categorical_cross_entropy = nn.CrossEntropyLoss()

    def forward(self, img, word,  ht_1=None, ct_1=None, mask=None, action_object_pairs=None):
        '''
            img: b, 3, h, w
            word: b, words
            word_mask: b, words
            mask: b, 1, h, w
        '''
        # padding mask used in decoder
        pad_mask = torch.zeros_like(word).masked_fill_(word == 0, 1).bool()
        # vis: C3 / C4 / C5
        # word: b, length, 1024
        # state: b, 1024
        vis = self.backbone.encode_image(img)
        word, state = self.backbone.encode_text(word)
        fq = self.neck(vis, state)
        b, c, h, w = fq.size()
        fq = self.decoder(fq, word, pad_mask)
        fq = fq.reshape(b, c, h, w)
        vl_features = torch.mean(fq, dim=(-1, -2))
        if self.training:
            split_counts = [ao.shape[0] for ao in action_object_pairs]
            vl_features_list = torch.split(vl_features, split_counts)
            action_losses, object_losses, eos_losses = [], [], []
        else:
            vl_features_list = [vl_features]
        action_object_logits = []
        action_logits_list, object_logits_list, eos_logits_list = [], [], []
        for seq_id, vl_features in enumerate(vl_features_list):
            if (ht_1 is not None) and (ct_1 is not None):
                out, (ht, ct) = self.action_object_predictor(vl_features.unsqueeze(0), (ht_1, ct_1))
            else:
                out, (ht, ct) = self.action_object_predictor(vl_features.unsqueeze(0))
            action_object_logits.append(out.squeeze(0))
            action_logits = self.action_classifier(out)
            object_logits = self.object_classifier(out)
            eos_logit = self.eos_classifier(out)
            if self.training:
                action_loss = self.categorical_cross_entropy(
                    action_logits.squeeze(0), action_object_pairs[seq_id][:, 0])
                object_loss = self.categorical_cross_entropy(
                    object_logits.squeeze(0), action_object_pairs[seq_id][:, 1])
                eos_loss = F.binary_cross_entropy_with_logits(
                    eos_logit.squeeze(0).squeeze(-1), action_object_pairs[seq_id][:, 2].float())
                action_losses.append(action_loss)
                object_losses.append(object_loss)
                eos_losses.append(eos_loss)

            action_logits_list.append(action_logits.clone().detach())
            object_logits_list.append(object_logits.clone().detach())
            eos_logits_list.append(eos_logit.clone().detach())
        action_object_logits = torch.cat(action_object_logits)
        # b, 1, 104, 104
        word_action_object = torch.cat([state, action_object_logits], dim=1)
        pred = self.proj(fq, word_action_object)
        if self.training:
            # resize mask
            if pred.shape[-2:] != mask.shape[-2:]:
                mask = F.interpolate(mask, pred.shape[-2:],
                                     mode='nearest').detach()
            mask_loss = F.binary_cross_entropy_with_logits(pred, mask)
            action_loss = torch.mean(torch.stack(action_losses))
            object_loss = torch.mean(torch.stack(object_losses))
            eos_loss = torch.mean(torch.stack(eos_losses))
            loss = (mask_loss) + (action_loss) + (object_loss) + (eos_loss)
            return pred.detach(), mask, mask_loss, action_loss, object_loss, eos_loss, loss, action_logits_list, object_logits_list, eos_logits_list
        else:
            return pred.detach(), action_logits_list, object_logits_list, eos_logits_list, ht.detach(), ct.detach()

class ModifiedAttentionPool2d(AttentionPool2d):
    def __init__(self,
                 spacial_dim: int,
                 embed_dim: int,
                 num_heads: int,
                 output_dim: int = None):
        super().__init__(
            spacial_dim,
            embed_dim,
            num_heads,
            output_dim
        )
        self.spacial_dim = spacial_dim
        self.connect = nn.Sequential(
            nn.Conv2d(embed_dim, output_dim, 1, stride=1, bias=False),
            nn.BatchNorm2d(output_dim))

    def resize_pos_embed(self, pos_embed, input_shpae):
        """Resize pos_embed weights.
        Resize pos_embed using bicubic interpolate method.
        Args:
            pos_embed (torch.Tensor): Position embedding weights.
            input_shpae (tuple): Tuple for (downsampled input image height,
                downsampled input image width).
            pos_shape (tuple): The resolution of downsampled origin training
                image.
            mode (str): Algorithm used for upsampling:
                ``'nearest'`` | ``'linear'`` | ``'bilinear'`` | ``'bicubic'`` |
                ``'trilinear'``. Default: ``'nearest'``
        Return:
            torch.Tensor: The resized pos_embed of shape [B, C, L_new]
        """
        assert pos_embed.ndim == 3, 'shape of pos_embed must be [B, L, C]'
        pos_h = pos_w = self.spacial_dim
        cls_token_weight = pos_embed[:, 0]
        pos_embed_weight = pos_embed[:, (-1 * pos_h * pos_w):]
        pos_embed_weight = pos_embed_weight.reshape(
            1, pos_h, pos_w, pos_embed.shape[2]).permute(0, 3, 1, 2)
        pos_embed_weight = F.interpolate(pos_embed_weight,
                                         size=input_shpae,
                                         align_corners=False,
                                         mode='bicubic')
        cls_token_weight = cls_token_weight.unsqueeze(1)
        pos_embed_weight = torch.flatten(pos_embed_weight, 2).transpose(1, 2)
        # pos_embed = torch.cat((cls_token_weight, pos_embed_weight), dim=1)
        return pos_embed_weight.transpose(-2, -1)

    def forward(self, x):
        B, C, H, W = x.size()
        res = self.connect(x)
        x = x.reshape(B, C, -1)  # NC(HW)
        # x = torch.cat([x.mean(dim=-1, keepdim=True), x], dim=-1)  # NC(1+HW)
        pos_embed = self.positional_embedding.unsqueeze(0)
        pos_embed = self.resize_pos_embed(pos_embed, (H, W))  # NC(HW)
        x = x + pos_embed.to(x.dtype)  # NC(HW)
        x = x.permute(2, 0, 1)  # (HW)NC
        x, _ = F.multi_head_attention_forward(
            query=x,
            key=x,
            value=x,
            embed_dim_to_check=x.shape[-1],
            num_heads=self.num_heads,
            q_proj_weight=self.q_proj.weight,
            k_proj_weight=self.k_proj.weight,
            v_proj_weight=self.v_proj.weight,
            in_proj_weight=None,
            in_proj_bias=torch.cat(
                [self.q_proj.bias, self.k_proj.bias, self.v_proj.bias]),
            bias_k=None,
            bias_v=None,
            add_zero_attn=False,
            dropout_p=0,
            out_proj_weight=self.c_proj.weight,
            out_proj_bias=self.c_proj.bias,
            use_separate_proj_weight=True,
            training=self.training,
            need_weights=False)
        x = x.permute(1, 2, 0).reshape(B, -1, H, W)
        x = x + res
        x = F.relu(x, True)

        return x


class ModifiedResnetWithPyramidOutput(ModifiedResNet):
    def __init__(
        self,
        layers,
        output_dim,
        heads,
        input_resolution=224,
        width=64
    ):
        super().__init__(
            layers,
            output_dim,
            heads,
            input_resolution=224,
            width=64)
        embed_dim = width * 32  # the ResNet feature dimension
        self.attnpool = ModifiedAttentionPool2d(
            input_resolution // 32, embed_dim,
            heads, output_dim)
        # Because one relu is enough
        self.relu1, self.relu2, self.relu3 = None, None, None
        self.relu =  nn.ReLU(inplace=True)

    def stem(self, x):
        for conv, bn in [(self.conv1, self.bn1), (self.conv2, self.bn2),
                        (self.conv3, self.bn3)]:
            x = self.relu(bn(conv(x)))
        x = self.avgpool(x)
        return x
    
    def forward(self, x):
        x = x.type(self.conv1.weight.dtype)
        x = self.stem(x)
        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)
        x4 = self.attnpool(x4)

        return (x2, x3, x4)


class ModifiedCLIP(CLIP):
    def __init__(
        self,
        embed_dim: int,
        # vision
        image_resolution: int,
        vision_layers: Union[Tuple[int, int, int, int], int],
        vision_width: int,
        vision_patch_size: int,
        # text
        context_length: int,
        vocab_size: int,
        transformer_width: int,
        transformer_heads: int,
        transformer_layers: int,
        txt_length: int):
        super().__init__(
            embed_dim, image_resolution, vision_layers, vision_width,
            vision_patch_size, context_length, vocab_size,
            transformer_width, transformer_heads, transformer_layers)
        vision_heads = vision_width * 32 // 64
        self.visual = ModifiedResnetWithPyramidOutput(
            layers=vision_layers,
            output_dim=embed_dim,
            heads=vision_heads,
            input_resolution=image_resolution,
            width=vision_width)
        attention_mask = self.build_attention_mask_(txt_length)
        self.transformer = Transformer(
            width=transformer_width,
            layers=transformer_layers,
            heads=transformer_heads,
            attn_mask=attention_mask)
        self.token_embedding.requires_grad_ = False
        self.initialize_parameters()

    def encode_text(self, text):
        x = self.token_embedding(text).type(
            self.dtype)  # [batch_size, n_ctx, d_model]

        x = x + self.positional_embedding.type(self.dtype)[:x.size(1)]
        x = x.permute(1, 0, 2)  # NLD -> LND
        x = self.transformer(x)
        x = x.permute(1, 0, 2)  # LND -> NLD
        x = self.ln_final(x).type(self.dtype)
        # x.shape = [batch_size, n_ctx, transformer.width]
        # take features from the eot embedding (eot_token is the highest number in each sequence)
        state = x[torch.arange(x.shape[0]),
                  text.argmax(dim=-1)] @ self.text_projection

        return x, state

    def build_attention_mask_(self, context_length):
        # lazily create causal attention mask, with full attention between the vision tokens
        # pytorch uses additive attention mask; fill with -inf
        mask = torch.empty(context_length, context_length)
        mask.fill_(float("-inf"))
        mask.triu_(1)  # zero out the lower diagonal
        return mask

    
def build_model(state_dict: dict, txt_length: int):
    vit = "visual.proj" in state_dict

    if vit:
        vision_width = state_dict["visual.conv1.weight"].shape[0]
        vision_layers = len([
            k for k in state_dict.keys()
            if k.startswith("visual.") and k.endswith(".attn.in_proj_weight")
        ])
        vision_patch_size = state_dict["visual.conv1.weight"].shape[-1]
        grid_size = round(
            (state_dict["visual.positional_embedding"].shape[0] - 1)**0.5)
        image_resolution = vision_patch_size * grid_size
    else:
        counts: list = [
            len(
                set(
                    k.split(".")[2] for k in state_dict
                    if k.startswith(f"visual.layer{b}")))
            for b in [1, 2, 3, 4]
        ]
        vision_layers = tuple(counts)
        vision_width = state_dict["visual.layer1.0.conv1.weight"].shape[0]
        output_width = round(
            (state_dict["visual.attnpool.positional_embedding"].shape[0] -
             1)**0.5)
        vision_patch_size = None
        assert output_width**2 + 1 == state_dict[
            "visual.attnpool.positional_embedding"].shape[0]
        image_resolution = output_width * 32

    embed_dim = state_dict["text_projection"].shape[1]
    context_length = state_dict["positional_embedding"].shape[0]
    vocab_size = state_dict["token_embedding.weight"].shape[0]
    transformer_width = state_dict["ln_final.weight"].shape[0]
    transformer_heads = transformer_width // 64
    transformer_layers = len(
        set(
            k.split(".")[2] for k in state_dict
            if k.startswith(f"transformer.resblocks")))
    # CLIP takes an extra argument of max sentence length
    model = ModifiedCLIP(
        embed_dim, image_resolution, vision_layers, vision_width,
        vision_patch_size, context_length, vocab_size,
        transformer_width, transformer_heads, transformer_layers, txt_length)
    for key in ["input_resolution", "context_length", "vocab_size"]:
        if key in state_dict:
            del state_dict[key]

    convert_weights(model)
    model.load_state_dict(state_dict, False)
    return model.eval()
