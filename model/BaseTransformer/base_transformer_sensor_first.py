import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os, math

from copy import deepcopy
from einops import rearrange, repeat

from model.BaseTransformer.spec_embed import SpecEmbedding
from model.BaseTransformer.attn import Attention


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


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(args.e_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(args.d_model, Attention(args)),
                        PreNorm(args.d_model, FeedForward(args.d_model, args.d_ff, dropout=args.dropout)),
                    ]
                )
            )

    def forward(self, x):
        attn_map_all = []
        for i, (attn, ff) in enumerate(self.layers):
            attn_x, attn_map = attn(x)
            x = attn_x + x
            x = ff(x) + x
            attn_map_all = attn_map if i == 0 else torch.cat((attn_map_all, attn_map), dim=0)

        return x, attn_map_all


class BaseTransformer(nn.Module):
    def __init__(self, cfg, args):
        super().__init__()

        self.num_control_features = cfg.NUM_CONTROL_FEATURES
        self.num_pred_features = cfg.NUM_PRED_FEATURES
        self.num_byproduct_features = cfg.NUM_BYPRODUCT_FEATURES
        self.num_target_features = cfg.NUM_TARGET_FEATURES
        self.num_all_features = cfg.NUM_ALL_FEATURES

        self.input_embedding = nn.Linear(args.in_len, args.d_model)
        self.positional_embedding = PositionalEmbedding(args.d_model)  # 絶対位置エンコーディング
        self.spec_embedding = SpecEmbedding(args.d_model)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_all_features + self.num_control_features, args.d_model))
        self.dropout = nn.Dropout(args.dropout)

        self.transformer = Transformer(args)

        self.generator = nn.Sequential(nn.LayerNorm(args.d_model), nn.Linear(args.d_model, self.num_pred_features))

    def forward(self, input, spec):
        # print('input.shape', input.shape)
        input = torch.permute(input, (0, 2, 1))
        # print(input.shape)
        x = self.input_embedding(input)
        # print('input_embedding', x.shape)
        # x += self.positional_embedding(x)
        # print('positional_embedding', x.shape)

        spec = self.spec_embedding(spec)

        x = torch.cat((x, spec), dim=1)

        # x += self.pos_embedding
        x = self.dropout(x)
        x, attn = self.transformer(x)
        # print('x.shape', x.shape)
        x = x.mean(dim=1)
        # print('x.shape mean', x.shape)
        x = self.generator(x)
        return x, attn
