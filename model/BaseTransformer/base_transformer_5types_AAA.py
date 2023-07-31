import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy
from einops import rearrange, repeat

from model.BaseTransformer.encode import AgentEncoding, TimeEncoding

from model.BaseTransformer.spec_embed import SpecEmbedding_9to4
from model.BaseTransformer.attn import AgentAwareAttention


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def clones(module, n):
    # produce N identical layers.
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


# classes
class InputEmbedding(nn.Module):
    def __init__(self, cfg, args):
        super(InputEmbedding, self).__init__()
        self.in_len = args.in_len

        self.agent_encoding = AgentEncoding(args.d_model, self.in_len)
        self.time_encoding = TimeEncoding(args.d_model, self.in_len)

        self.comp = nn.Linear(14, args.d_model)
        self.cond = nn.Linear(2, args.d_model)
        self.evap = nn.Linear(19, args.d_model)
        self.chil = nn.Linear(14, args.d_model)
        self.acds = nn.Linear(1, args.d_model)

    def forward(self, x):
        comp = self.comp(torch.cat((x[:, :, 0].unsqueeze(2), x[:, :, 9:15], x[:, :, 30:36], x[:, :, 49].unsqueeze(2)), dim=2))
        cond = self.cond(x[:, :, 1:3])
        evap = self.evap(torch.cat((x[:, :, 3:5], x[:, :, 7].unsqueeze(2), x[:, :, 15:30], x[:, :, 47].unsqueeze(2)), dim=2))
        chil = self.chil(torch.cat((x[:, :, 5:7], x[:, :, 8].unsqueeze(2), x[:, :, 36:46], x[:, :, 48].unsqueeze(2)), dim=2))
        acds = self.acds(x[:, :, 46].unsqueeze(2))

        x = torch.cat((comp, cond, evap, chil, acds), dim=1)

        x += self.agent_encoding(x)
        x += self.time_encoding(x)

        """
        img_path = os.path.join("img", "inp_5types", "encoding")
        os.makedirs(img_path, exist_ok=True)
        # time encoding visualization
        te = self.time_encoding(x).to("cpu").detach().numpy().copy()
        print("te", te.shape)
        fig = plt.figure()
        plt.imshow(te[0])
        plt.colorbar()
        plt.savefig(f"img/inp_5types/encoding/time_encoding_input_5types_lookback{self.in_len}.png")
        # agent encoding visualization
        ae = self.agent_encoding(x).to("cpu").detach().numpy().copy()
        print("ae", ae.shape)
        fig = plt.figure()
        plt.imshow(ae[0])
        plt.colorbar()
        plt.savefig(f"img/inp_5types/encoding/agent_encoding_input_5types_lookback{self.in_len}.png")
        """

        return x


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
    def __init__(self, args, num_agent, num_control_features):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(args.e_layers):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(args.d_model, AgentAwareAttention(args, num_agent, num_control_features)),
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

        self.num_control_features = 4
        self.num_pred_features = cfg.NUM_PRED_FEATURES
        self.num_agent = 5

        self.input_embedding = InputEmbedding(cfg, args)
        self.spec_embedding = SpecEmbedding_9to4(args.d_model)

        self.dropout = nn.Dropout(args.dropout)

        self.transformer = Transformer(args, self.num_agent, self.num_control_features)

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
