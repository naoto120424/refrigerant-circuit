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
class AgentEncoding(nn.Module):
    def __init__(self, d_model, look_back, max_len=5000):
        super(AgentEncoding, self).__init__()
        # Compute the positional encodings once in log space.
        self.look_back = look_back
        ae = torch.zeros(max_len, d_model).float()
        ae.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        ae[:, 0::2] = torch.sin(position * div_term)
        ae[:, 1::2] = torch.cos(position * div_term)

        ae = ae.unsqueeze(0)
        self.register_buffer("ae", ae)

    def forward(self, x):
        # print(x.shape)
        num_a = int(x.size(1) / self.look_back)
        # print(num_a)
        ae = self.ae[:, :num_a]
        # print(ae.shape)
        ae = ae.repeat_interleave(self.look_back, dim=1)
        # print(ae.shape)
        return ae[:, : x.size(1)]


class TimeEncoding(nn.Module):
    def __init__(self, d_model, look_back, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.look_back = look_back
        te = torch.zeros(max_len, d_model).float()
        te.requires_grad = False
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        te[:, 0::2] = torch.sin(position * div_term)
        te[:, 1::2] = torch.cos(position * div_term)
        te = te.unsqueeze(0)
        self.register_buffer("te", te)

    def forward(self, x):
        # print(x.shape)
        te = self.te[:, : self.look_back]
        num_a = int(x.size(1) / self.look_back)
        # print(num_a)
        te = te.repeat(1, num_a, 1)
        # print(ae.shape)
        return te[:, : x.size(1)]


class InputEmbedding(nn.Module):
    def __init__(self, cfg, args):
        super(InputEmbedding, self).__init__()
        self.in_len = args.in_len

        self.linear = nn.Linear(1, args.d_model)

        self.time_encoding = TimeEncoding(args.d_model, self.in_len)
        self.agent_encoding = AgentEncoding(args.d_model, self.in_len)

    def forward(self, x):
        batch, ts_len, ts_dim = x.shape

        x = rearrange(x, "b t d -> (b t d) 1")
        x_embed = self.linear(x)
        x_embed = rearrange(x_embed, "(b td) d_model -> b td d_model", b=batch)

        x_embed += self.agent_encoding(x_embed)
        x_embed += self.time_encoding(x_embed)

        """
        img_path = os.path.join("img", "inp_individually", "encoding")
        os.makedirs(img_path, exist_ok=True)
        # positional encoding visualization
        te = self.time_encoding(x_embed).to("cpu").detach().numpy().copy()
        print("te", te.shape)
        fig = plt.figure()
        plt.imshow(te[0])
        plt.colorbar()
        plt.savefig(f"img/inp_individually/encoding/time_encoding_input_individually_lookback{ts_len}.png")

        # agent encoding visualization
        ae = self.agent_encoding(x_embed).to("cpu").detach().numpy().copy()
        print("ae", ae.shape)
        fig = plt.figure()
        plt.imshow(ae[0])
        plt.colorbar()
        plt.savefig(f"img/inp_individually/encoding/agent_encoding_input_individually_lookback{ts_len}.png")
        """

        return x_embed


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
        self.in_len = args.in_len

        self.input_embedding = InputEmbedding(cfg, args)
        self.spec_embedding = SpecEmbedding(args.d_model)

        self.dropout = nn.Dropout(args.dropout)

        self.transformer = Transformer(args)

        self.generator = nn.Sequential(nn.LayerNorm(args.d_model), nn.Linear(args.d_model, self.num_pred_features))

    def forward(self, input, spec):
        x = self.input_embedding(input)

        spec = self.spec_embedding(spec)

        x = torch.cat((x, spec), dim=1)

        x = self.dropout(x)
        x, attn = self.transformer(x)
        x = x.mean(dim=1)
        x = self.generator(x)
        return x, attn
