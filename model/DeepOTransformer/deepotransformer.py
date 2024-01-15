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
        """
        img_path = os.path.join("img", "inp_normal", "encoding")
        os.makedirs(img_path, exist_ok=True)
        pe = self.pe[:, : x.size(1)].to("cpu").detach().numpy().copy()
        fig = plt.figure()
        plt.imshow(pe[0])
        plt.colorbar()
        plt.savefig(f"img/inp_normal/encoding/time_encoding_input_norm_lookback{x.size(1)}.png")
        """
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


class DeepOTransformer(nn.Module):
    def __init__(self, cfg, args):
        super().__init__()

        self.num_control_features = cfg.NUM_CONTROL_FEATURES
        self.num_pred_features = cfg.NUM_PRED_FEATURES
        self.num_byproduct_features = cfg.NUM_BYPRODUCT_FEATURES
        self.num_target_features = cfg.NUM_TARGET_FEATURES
        self.num_all_features = cfg.NUM_ALL_FEATURES
        self.in_len = args.in_len

        self.input_embedding = nn.Linear(self.num_all_features, args.d_model)
        self.positional_embedding = PositionalEmbedding(args.d_model)  # 絶対位置エンコーディング
        self.spec_embedding = SpecEmbedding(args.d_model)

        self.dropout = nn.Dropout(args.dropout)

        self.transformer = Transformer(args)
        
        self.branch = nn.Linear(self.in_len+self.num_control_features, self.num_pred_features)
        
        # self.trunk = nn.Sequential(
        #     nn.Linear(1, args.d_model),
        #     nn.ReLU(),
        #     nn.Linear(args.d_model, args.d_model),
        #     nn.ReLU(),
        #     nn.Linear(args.d_model, args.d_model),
        #     nn.ReLU(),
        #     nn.Dropout(args.dropout),
        # )
        
        self.trunk = nn.ModuleList([])
        self.trunk.append(
            nn.ModuleList(
                [
                    nn.Linear(1, args.trunk_d_model),
                    nn.ReLU()
                ]
            )
        )
        for _ in range(args.trunk_layers-2):
            self.trunk.append(
                nn.ModuleList(
                    [
                        nn.Linear(args.trunk_d_model, args.trunk_d_model),
                        nn.ReLU(),
                    ]
                )
            )
        self.trunk.append(
            nn.ModuleList(
                [
                    nn.Linear(args.trunk_d_model, args.d_model),
                    nn.ReLU(),
                ]
            )
        )
        self.trunk_dropout = nn.Dropout(args.dropout)

    def forward(self, input, spec, timedata):
        x = self.input_embedding(input)
        x += self.positional_embedding(x)

        spec = self.spec_embedding(spec)

        x = torch.cat((x, spec), dim=1)

        x = self.dropout(x)
        x, attn = self.transformer(x)
        x = self.branch(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        y = timedata.repeat(1, self.num_pred_features).unsqueeze(2)
        
        for linear, relu in self.trunk:
            y = linear(y)
            y = relu(y)
        
        # print(f"x.shape: {x.shape}, y.shape: {y.shape}")
        x = torch.sum(x * y, dim=-1, keepdim=True)
        x = x.squeeze(2)
        
        return x, attn
