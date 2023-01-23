import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os, math

from copy import deepcopy
from einops import rearrange, repeat


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def clones(module, n):
    # produce N identical layers.
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


# classes
class PositionalEmbedding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEmbedding, self).__init__()
        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer("pe", pe)

    def forward(self, x):
        return self.pe[:, : x.size(1)]


class SpecEmbedding(nn.Module):
    def __init__(self, dim, num_control_features):
        super().__init__()
        self.spec_emb_list = clones(nn.Linear(1, dim), num_control_features)

    def forward(self, spec):
        spec = torch.unsqueeze(spec, 1)  # bx9 -> bx1x9
        spec = torch.unsqueeze(spec, 1)  # bx1x9 -> bx1x1x9

        for i, spec_embedding in enumerate(self.spec_emb_list):
            if i == 0:
                spec_emb_all = spec_embedding(spec[:, :, :, i])
            else:
                spec_emb_all = torch.cat((spec_emb_all, spec_embedding(spec[:, :, :, i])), dim=1)

        return spec_emb_all


class Transformer(nn.Module):
    def __init__(self, cfg, args):
        super().__init__()

        self.num_control_features = cfg.NUM_CONTROL_FEATURES
        self.num_pred_features = cfg.NUM_PRED_FEATURES
        self.num_byproduct_features = cfg.NUM_BYPRODUCT_FEATURES
        self.num_target_features = cfg.NUM_TARGET_FEATURES
        self.num_all_features = cfg.NUM_ALL_FEATURES

        self.input_embedding = nn.Linear(self.num_all_features, args.dim)
        self.positional_embedding = PositionalEmbedding(args.dim)  # 絶対位置エンコーディング

        self.gt_embedding = nn.Linear(self.num_pred_features, args.dim)
        self.spec_embedding = SpecEmbedding(args.dim, self.num_control_features)

        self.dropout = nn.Dropout(args.emb_dropout)

        self.transformer = nn.Transformer(
            d_model=args.dim,
            nhead=args.heads,
            num_encoder_layers=args.depth,
            num_decoder_layers=args.depth,
            dim_feedforward=args.fc_dim,
            dropout=args.dropout,
            batch_first=True,
        )

        self.generator = nn.Sequential(nn.LayerNorm(args.dim), nn.Linear(args.dim, self.num_pred_features))

    def forward(self, input, spec, gt):
        x = self.input_embedding(input)
        x += self.positional_embedding(x)

        gt = torch.unsqueeze(gt, 1)
        y = self.gt_embedding(gt)

        spec = self.spec_embedding(spec)

        x = torch.cat((x, spec), dim=1)

        x = self.dropout(x)
        x = self.transformer(x, y)
        x = x.mean(dim=1)
        x = self.generator(x)

        return x
