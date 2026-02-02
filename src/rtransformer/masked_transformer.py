"""
 Copyright (c) 2018, salesforce.com, inc.
 All rights reserved.
 SPDX-License-Identifier: BSD-3-Clause
 For full license text, see the LICENSE file in the repo root or https://opensource.org/licenses/BSD-3-Clause

References:
    https://github.com/salesforce/densecap/blob/master/model/transformer.py

Modified by Jie Lei
"""

import torch
from torch import nn
from torch.nn import functional as F
from torchvision import models

import math
import numpy as np
from src.rtransformer.model import LabelSmoothingLoss


INF = 1e10


# def positional_encodings_like(x, t=None):
#     device = x.get_device() if x.is_cuda else 'cpu'
#     if t is None:
#         positions = torch.arange(0, x.size(1), device=device).float()
#         # if x.is_cuda:
#         #    positions = positions.cuda(x.get_device())
#     else:
#         positions = t
#     encodings = torch.zeros(*x.size()[1:], device=device)
#     # if x.is_cuda:
#     #     encodings = encodings.cuda(x.get_device())

#     for channel in range(x.size(-1)):
#         if channel % 2 == 0:
#             encodings[:, channel] = torch.sin(positions / 10000 ** (channel / x.size(2)))
#         else:
#             encodings[:, channel] = torch.cos(positions / 10000 ** ((channel - 1) / x.size(2)))
#     return encodings

def positional_encodings_like(x: torch.Tensor, t: torch.Tensor = None):
    """
    Generate sinusoidal positional encodings like tensor `x` (no batch dim assumed).
    Args:
        x: Tensor of shape (T, D) or (N, T, D)
        t: Optional manual position indices (shape: T)
    Returns:
        Tensor of shape (T, D) or (N, T, D), same as x without batch dimension support.
    """
    device = x.device
    T = x.size(-2)  # sequence length
    D = x.size(-1)  # embedding dim

    if t is None:
        positions = torch.arange(T, device=device, dtype=torch.float32).unsqueeze(1)  # (T, 1)
    else:
        positions = t.to(device=device, dtype=torch.float32).unsqueeze(1)             # (T, 1)

    div_term = torch.exp(torch.arange(0, D, 2, device=device, dtype=torch.float32) * (-math.log(10000.0) / D))  # (D//2,)

    pe = torch.zeros(T, D, device=device)
    pe[:, 0::2] = torch.sin(positions * div_term)   # even dims
    pe[:, 1::2] = torch.cos(positions * div_term)   # odd dims

    return pe  # shape: (T, D)


class LayerNorm(nn.Module):
    def __init__(self, d_model, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(d_model))
        self.beta = nn.Parameter(torch.zeros(d_model))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta


class ResidualBlock(nn.Module):
    def __init__(self, layer, d_model, drop_ratio):
        super(ResidualBlock, self).__init__()
        self.layer = layer
        self.dropout = nn.Dropout(drop_ratio)
        self.layernorm = LayerNorm(d_model)

    def forward(self, *x):
        return self.layernorm(x[0] + self.dropout(self.layer(*x)))


class Attention(nn.Module):

    def __init__(self, d_key, drop_ratio, causal):
        super(Attention, self).__init__()
        self.scale = math.sqrt(d_key)
        self.dropout = nn.Dropout(drop_ratio)
        self.causal = causal
    
    def forward(self, query, key, value):
        dot_products = torch.bmm(query, key.transpose(1, 2))  # (B, Tq, Tk)

        if query.dim() == 3 and self.causal:
            # 创建 causal mask
            seq_len = key.size(1)
            mask = torch.triu(
                torch.full((seq_len, seq_len), float("-inf"), device=key.device, dtype=dot_products.dtype),
                diagonal=1
            )  # shape: (Tk, Tk)
            dot_products = dot_products + mask.unsqueeze(0)  # broadcast to (B, Tq, Tk)

        attn_weights = F.softmax(dot_products / self.scale, dim=-1)
        attn_weights = self.dropout(attn_weights)

        return torch.bmm(attn_weights, value)

    # def forward(self, query, key, value):
    #     dot_products = torch.bmm(query, key.transpose(1, 2))
    #     if query.dim() == 3 and (self is None or self.causal):
    #         tri = torch.ones(key.size(1), key.size(1)).triu(1) * INF
    #         if key.is_cuda:
    #             tri = tri.cuda(key.get_device())
    #         dot_products.data.sub_(tri.unsqueeze(0))

    #     return torch.bmm(self.dropout(F.softmax(dot_products / self.scale, dim=-1)), value)


class MultiHead(nn.Module):
    def __init__(self, d_key, d_value, n_heads, drop_ratio, causal=False):
        super(MultiHead, self).__init__()
        self.attention = Attention(d_key, drop_ratio, causal=causal)
        self.wq = nn.Linear(d_key, d_key, bias=False)
        self.wk = nn.Linear(d_key, d_key, bias=False)
        self.wv = nn.Linear(d_value, d_value, bias=False)
        self.wo = nn.Linear(d_value, d_key, bias=False)
        self.n_heads = n_heads

    def forward(self, query, key, value):
        query, key, value = self.wq(query), self.wk(key), self.wv(value)
        query, key, value = (
            x.chunk(self.n_heads, -1) for x in (query, key, value))
        return self.wo(torch.cat([self.attention(q, k, v)
                       for q, k, v in zip(query, key, value)], -1))


class FeedForward(nn.Module):
    def __init__(self, d_model, d_hidden):
        super(FeedForward, self).__init__()
        self.linear1 = nn.Linear(d_model, d_hidden)
        self.linear2 = nn.Linear(d_hidden, d_model)

    def forward(self, x):
        return self.linear2(F.relu(self.linear1(x)))


class EncoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, drop_ratio):
        super(EncoderLayer, self).__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio, causal=False),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x):
        return self.feedforward(self.selfattn(x, x, x))


class DecoderLayer(nn.Module):
    def __init__(self, d_model, d_hidden, n_heads, drop_ratio, casual_sa=True):
        super(DecoderLayer, self).__init__()
        self.selfattn = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio, causal=casual_sa),
            d_model, drop_ratio)
        self.attention = ResidualBlock(
            MultiHead(d_model, d_model, n_heads, drop_ratio),
            d_model, drop_ratio)
        self.feedforward = ResidualBlock(FeedForward(d_model, d_hidden),
                                         d_model, drop_ratio)

    def forward(self, x, encoding, without_corss=False):
        """
        Args:
            x: (N, Lt, D)
            encoding: (N, Lv, D)
        """
        x = self.selfattn(x, x, x)  # (N, Lt, D)
        if without_corss:
            return self.feedforward(x)
        return self.feedforward(self.attention(x, encoding, encoding))  # (N, Lt, D)


class Encoder(nn.Module):
    def __init__(self, vfeat_size, d_model, d_hidden, n_layers, n_heads, drop_ratio):
        super(Encoder, self).__init__()
        self.video_embeddings = nn.Sequential(
            LayerNorm(vfeat_size),
            nn.Dropout(drop_ratio),
            nn.Linear(vfeat_size, d_model)
        )
        self.layers = nn.ModuleList(
            [EncoderLayer(d_model, d_hidden, n_heads, drop_ratio)
             for i in range(n_layers)])
        self.dropout = nn.Dropout(drop_ratio)

    def forward(self, x, mask=None):
        """

        Args:
            x: (N, Lv, Dv)
            mask: (N, Lv)

        Returns:

        """
        x = self.video_embeddings(x)  # (N, Lv, D)
        x = x + positional_encodings_like(x)
        x = self.dropout(x)
        mask.unsqueeze_(-1)
        if mask is not None:
            x = x*mask
        encoding = []
        for layer in self.layers:
            x = layer(x)
            if mask is not None:
                x = x*mask
            encoding.append(x)
        return encoding


class VideoFeatureExtractor2D(nn.Module):
    def __init__(self, frame_num, in_resolution, in_channels, out_dim, dropout=0.1):
        super(VideoFeatureExtractor2D, self).__init__()
        self.frame_num = frame_num
        self.in_resolution = in_resolution
        self.in_channels = in_channels
        self.out_dim = out_dim
        self.dropout = dropout
        self.resnet50 = models.resnet50(pretrained=False)
        self.resnet50.fc = nn.Linear(self.resnet50.fc.in_features, out_dim)
    
    def forward(self, video_clip):
        features = []
        for i in range(video_clip.size(2)):
            x = self.resnet50(video_clip[:, :, i, :, :].squeeze(2))
            features.append(x)
        
        features = torch.mean(torch.stack(features, dim=1), dim=1)
        return features


class Decoder(nn.Module):
    def __init__(self, d_model, d_hidden, vocab_size, n_layers, n_heads,
                 drop_ratio, casual_sa=True, return_cls=False):
        super(Decoder, self).__init__()
        self.layers = nn.ModuleList(
            [DecoderLayer(d_model, d_hidden, n_heads, drop_ratio, casual_sa)
             for i in range(n_layers)])
        self.out = nn.Linear(d_model, vocab_size)
        self.dropout = nn.Dropout(drop_ratio)
        self.d_model = d_model
        self.d_out = vocab_size

        # # Add a learnable class token for video text contrastive learning
        if not casual_sa and return_cls:
            self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        else:
            self.cls = None

    def forward(self, x, encoding):
        """
        Args:
            x: (N, Lt)
            encoding: [(N, Lv, D), ] * num_hidden_layers

        """
        x = F.embedding(x, self.out.weight * math.sqrt(self.d_model))  # (N, Lt, D)

        if self.cls is not None:
            x_cls = self.cls.expand(x.size(0), 1, self.d_model)  # (N, 1, D)
            x_self = torch.cat([x_cls, x.clone()], 1)  # (N, Lt+1, D)
            x_self = x_self + positional_encodings_like(x_self)  # (N, Lt+1, D)
            x_self = self.dropout(x_self)  # (N, Lt+1, D)

        x = x + positional_encodings_like(x)  # (N, Lt, D)
        x = self.dropout(x)  # (N, Lt, D)
        for layer, enc in zip(self.layers, encoding):
            if self.cls is None:
                x = layer(x, enc)  # (N, Lt, D)
            else:
                x = layer(x, enc, without_corss=False)
                x_self = layer(x_self, enc, without_corss=True)
        
        x_cls = x_self[:, 0, :] if self.cls is not None else None  # (N, D)
        return x, x_cls
        # return x  # (N, Lt, D) at last layer


class MTransformer(nn.Module):
    def __init__(self, config):
        super(MTransformer, self).__init__()
        self.config = config
        vfeat_size = config.video_feature_size
        d_model = config.hidden_size  # 1024
        d_hidden = config.intermediate_size  # 2048
        n_layers = config.num_hidden_layers  # 6
        n_heads = config.num_attention_heads  # 8
        drop_ratio = config.hidden_dropout_prob  # 0.1
        self.vocab_size = config.vocab_size
        self.encoder = Encoder(vfeat_size, d_model, d_hidden, n_layers,
                               n_heads, drop_ratio)
        # self.encoder = nn.Identity() # Remove the visual encoder
        self.decoder = Decoder(d_model, d_hidden, self.vocab_size,
                               n_layers, n_heads, drop_ratio)
        self.loss_func = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=-1) \
            if "label_smoothing" in config and config.label_smoothing > 0 else nn.CrossEntropyLoss(ignore_index=-1)

    def encode(self, video_features, video_masks):
        """
        Args:
            video_features: (N, Lv, Dv)
            video_masks: (N, Lv)  with 1 indicates valid bits
        """
        return self.encoder(video_features, video_masks)
    # def encode(self, video_features, video_masks):
    #     """
    #     Args:
    #         video_features: (B, T, C, H, W)
    #         video_masks: not required, leave here to maintain a common API with untied model
    #     """
    #     return self.encoder(video_features)

    def decode(self, text_input_ids, text_masks, text_input_labels, encoder_outputs, video_masks):
        """
        Args:
            text_input_ids: (N, Lt)
            text_masks: (N, Lt)  with 1 indicates valid bits,
            text_input_labels: (N, Lt)  with `-1` on ignored positions
            encoder_outputs: (N, Lv, D)
            video_masks: not used, leave here to maintain a common API with untied model
        """
        # the triangular mask is generated and applied inside the attention module
        h, h_cls = self.decoder(text_input_ids, encoder_outputs)  # (N, Lt, D)
        prediction_scores = self.decoder.out(h)  # (N, Lt, vocab_size)
        caption_loss = self.loss_func(prediction_scores.view(-1, self.config.vocab_size),
                                      text_input_labels.view(-1))  # float
        return caption_loss, prediction_scores, h_cls

    def forward(self, video_features, video_masks, text_input_ids, text_masks, text_input_labels):
        """
        Args:
            video_features: (N, Lv, Dv)
            video_masks: (N, Lv)  with 1 indicates valid bits
            text_input_ids: (N, Lt)
            text_masks: (N, Lt)  with 1 indicates valid bits
            text_input_labels: (N, Lt)  with `-1` on ignored positions  (in some sense duplicate with text_masks)
        """
        encoder_layer_outputs = self.encode(video_features, video_masks)  # [(N, Lv, D), ] * num_hidden_layers
        # encoder_layer_outputs = video_features
        caption_loss, prediction_scores, h_cls = self.decode(
            text_input_ids, text_masks, text_input_labels, encoder_layer_outputs, None)  # float, (N, Lt, vocab_size)
        return caption_loss, prediction_scores, h_cls


class MTransformerWithoutVisionEncoder(nn.Module):
    def __init__(self, config):
        super(MTransformerWithoutVisionEncoder, self).__init__()
        self.config = config
        d_model = config.hidden_size  # 1024
        d_hidden = config.intermediate_size  # 2048
        n_layers = config.num_hidden_layers  # 6
        n_heads = config.num_attention_heads  # 8
        drop_ratio = config.hidden_dropout_prob  # 0.1
        self.vocab_size = config.vocab_size
        if self.config.__contains__("training_stage") and self.config.training_stage == 1:
            self.decoder = Decoder(d_model, d_hidden, self.vocab_size,
                                n_layers, n_heads, drop_ratio, casual_sa=False, return_cls=config.vtc)
        else:
            self.decoder = Decoder(d_model, d_hidden, self.vocab_size,
                                n_layers, n_heads, drop_ratio, casual_sa=True)
            
        self.loss_func = LabelSmoothingLoss(config.label_smoothing, config.vocab_size, ignore_index=-1) \
            if "label_smoothing" in config and config.label_smoothing > 0 else nn.CrossEntropyLoss(ignore_index=-1)

    def decode(self, text_input_ids, text_masks, text_input_labels, encoder_outputs, video_masks):
        """
        Args:
            text_input_ids: (N, Lt)
            text_masks: (N, Lt)  with 1 indicates valid bits,
            text_input_labels: (N, Lt)  with `-1` on ignored positions
            encoder_outputs: (N, Lv, D)
            video_masks: not used, leave here to maintain a common API with untied model
        """
        # the triangular mask is generated and applied inside the attention module
        h, h_cls = self.decoder(text_input_ids, encoder_outputs)  # (N, Lt, D)
        prediction_scores = self.decoder.out(h)  # (N, Lt, vocab_size)
        caption_loss = self.loss_func(prediction_scores.view(-1, self.config.vocab_size),
                                      text_input_labels.view(-1))  # float
        return caption_loss, prediction_scores, h_cls

    def forward(self, video_features, text_input_ids, text_masks, text_input_labels):
        """
        Args:
            video_features: (N, Lv, Dv)
            video_masks: (N, Lv)  with 1 indicates valid bits
            text_input_ids: (N, Lt)
            text_masks: (N, Lt)  with 1 indicates valid bits
            text_input_labels: (N, Lt)  with `-1` on ignored positions  (in some sense duplicate with text_masks)
        """
        encoder_layer_outputs = [video_features] * self.config.num_hidden_layers  # [(N, Lv, D), ] * num_hidden_layers
        caption_loss, prediction_scores, h_cls = self.decode(
            text_input_ids, text_masks, text_input_labels, encoder_layer_outputs, None)  # float, (N, Lt, vocab_size)
        return caption_loss, prediction_scores, h_cls
    