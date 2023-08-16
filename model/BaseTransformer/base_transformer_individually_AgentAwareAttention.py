import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os, math

from copy import deepcopy
from einops import rearrange, repeat

from model.BaseTransformer.spec_embed import SpecEmbedding
from model.BaseTransformer.encode import AgentEncoding, TimeEncoding
from model.BaseTransformer.attn import AgentAwareAttention


# classes
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

        self.num_control_features = 9
        self.num_pred_features = cfg.NUM_PRED_FEATURES
        self.num_byproduct_features = cfg.NUM_BYPRODUCT_FEATURES
        self.num_target_features = cfg.NUM_TARGET_FEATURES
        self.num_all_features = cfg.NUM_ALL_FEATURES
        self.num_agent = self.num_all_features

        self.input_embedding = InputEmbedding(cfg, args)
        self.spec_embedding = SpecEmbedding(args.d_model)

        self.dropout = nn.Dropout(args.dropout)

        self.transformer = Transformer(args, self.num_agent, self.num_control_features)  # Transformer(args, self.num_agent, self.num_control_features)

        self.generator = nn.Sequential(nn.LayerNorm(args.d_model), nn.Linear(args.d_model, self.num_pred_features))
        self.generator2 = nn.Sequential(nn.LayerNorm(args.in_len * self.num_agent + self.num_control_features), nn.Linear(args.in_len * self.num_agent + self.num_control_features, self.num_pred_features))

    def forward(self, input, spec):
        x = self.input_embedding(input)

        spec = self.spec_embedding(spec)

        x = torch.cat((x, spec), dim=1)

        x = self.dropout(x)
        x, attn = self.transformer(x)

        x = x.mean(dim=1)
        x = self.generator(x)

        return x, attn
