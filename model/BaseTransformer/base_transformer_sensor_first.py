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


class Attention(nn.Module):
    def __init__(self, args):
        super().__init__()
        inner_dim = args.dim_head * args.heads
        project_out = not (args.heads == 1 and args.dim_head == args.dim)

        self.heads = args.heads
        self.scale = args.dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(args.dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, args.dim), nn.Dropout(args.dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out), attn


class Transformer(nn.Module):
    def __init__(self, args):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(args.depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(args.dim, Attention(args)),
                        PreNorm(args.dim, FeedForward(args.dim, args.fc_dim, dropout=args.dropout)),
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

        self.input_embedding = nn.Linear(args.look_back, args.dim)
        self.positional_embedding = PositionalEmbedding(args.dim)  # 絶対位置エンコーディング
        self.spec_embedding = SpecEmbedding(args.dim, self.num_control_features)

        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_all_features + self.num_control_features, args.dim))
        self.dropout = nn.Dropout(args.emb_dropout)

        self.transformer = Transformer(args)

        self.generator = nn.Sequential(nn.LayerNorm(args.dim), nn.Linear(args.dim, self.num_pred_features))

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
