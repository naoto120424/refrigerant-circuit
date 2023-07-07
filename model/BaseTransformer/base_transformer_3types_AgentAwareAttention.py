import torch
from torch import nn
import matplotlib.pyplot as plt
import numpy as np
import os
from copy import deepcopy
from einops import rearrange, repeat

from model.BaseTransformer.encode import AgentEncoding, TimeEncoding
from model.BaseTransformer.spec_embed import SpecEmbedding


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
        self.look_back = args.look_back
        self.num_control_features = cfg.NUM_CONTROL_FEATURES
        self.num_byproduct_features = cfg.NUM_BYPRODUCT_FEATURES
        self.num_target_features = cfg.NUM_TARGET_FEATURES

        self.control_embedding = nn.Linear(self.num_control_features, args.dim)
        self.byproduct_embedding = nn.Linear(self.num_byproduct_features, args.dim)
        self.target_embedding = nn.Linear(self.num_target_features, args.dim)

        self.agent_encoding = AgentEncoding(args.dim, self.look_back)
        self.time_encoding = TimeEncoding(args.dim, self.look_back)

    def forward(self, x):
        control = self.control_embedding(x[:, :, : self.num_control_features])
        # control += self.positional_embedding(control)
        print("control embedding: ", control.shape)
        byproduct = self.byproduct_embedding(x[:, :, self.num_control_features : self.num_control_features + self.num_byproduct_features])
        # byproduct += self.positional_embedding(byproduct)
        print("byproduct embedding: ", byproduct.shape)
        target = self.target_embedding(x[:, :, self.num_control_features + self.num_byproduct_features :])
        # target += self.positional_embedding(target)
        print("target embedding: ", target.shape)

        x = torch.cat([control, byproduct, target], dim=1)
        x += self.agent_encoding(x)
        x += self.time_encoding(x)

        """
        img_path = os.path.join("img", "inp_3types", "encoding")
        os.makedirs(img_path, exist_ok=True)
        # time encoding visualization
        te = self.time_encoding(x).to("cpu").detach().numpy().copy()
        print("te", te.shape)
        fig = plt.figure()
        plt.imshow(te[0])
        plt.colorbar()
        plt.savefig(f"img/inp_3types/encoding/time_encoding_input_3types_lookback{self.look_back}.png")
        # agent encoding visualization
        ae = self.agent_encoding(x).to("cpu").detach().numpy().copy()
        print("ae", ae.shape)
        fig = plt.figure()
        plt.imshow(ae[0])
        plt.colorbar()
        plt.savefig(f"img/inp_3types/encoding/agent_encoding_input_3types_lookback{self.look_back}.png")
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


class AgentAwareAttention(nn.Module):
    def __init__(self, args, num_agent=200, num_control_features=9):
        super().__init__()
        self.num_agent = num_agent
        self.num_control_features = num_control_features
        self.look_back = args.look_back
        inner_dim = args.dim_head * args.heads
        project_out = not (args.heads == 1 and args.dim_head == args.dim)

        self.heads = args.heads
        self.scale = args.dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(args.dim, inner_dim * 3, bias=False)
        self.to_qk_self = nn.Linear(args.dim, inner_dim * 2, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, args.dim), nn.Dropout(args.dropout)) if project_out else nn.Identity()

        attn_mask = torch.eye(self.num_agent)
        attn_mask = attn_mask.repeat_interleave(self.look_back, dim=1)
        attn_mask = attn_mask.repeat_interleave(self.look_back, dim=0)
        attn_mask = torch.cat([attn_mask, torch.zeros(attn_mask.size(0), self.num_control_features)], dim=1)
        attn_mask = torch.cat([attn_mask, torch.zeros(self.num_control_features, attn_mask.size(1))], dim=0)
        self.attn_mask = attn_mask.unsqueeze(0).unsqueeze(0)

        """
        img_path = os.path.join("img", "inp_3types", "attention")
        os.makedirs(img_path, exist_ok=True)
        fig = plt.figure()
        plt.imshow(attn_mask, cmap="Blues")
        plt.colorbar()
        plt.savefig(f"img/inp_3types/attention/attention_mask_input_3types_lookback{self.look_back}.png")
        """

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        qk_self = self.to_qk_self(x).chunk(2, dim=-1)
        q_self, k_self = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qk_self)

        attn_mask = self.attn_mask
        attn_mask = attn_mask.to(x.device)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        dots_self = torch.matmul(q_self, k_self.transpose(-1, -2)) * self.scale
        dots_all = attn_mask * dots_self + (1 - attn_mask) * dots

        attn = self.attend(dots_all)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")

        return self.to_out(out), attn


class Transformer(nn.Module):
    def __init__(self, args, num_agent, num_control_features):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(args.depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(args.dim, AgentAwareAttention(args, num_agent, num_control_features)),
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
        self.num_agent = 3

        self.input_embedding = InputEmbedding(cfg, args)
        self.spec_embedding = SpecEmbedding(args.dim)

        self.dropout = nn.Dropout(args.emb_dropout)

        self.transformer = Transformer(args, self.num_agent, self.num_control_features)

        self.generator = nn.Sequential(nn.LayerNorm(args.dim), nn.Linear(args.dim, self.num_pred_features))

    def forward(self, input, spec):
        x = self.input_embedding(input)

        spec = self.spec_embedding(spec)

        x = torch.cat((x, spec), dim=1)

        x = self.dropout(x)
        x, attn = self.transformer(x)
        x = x.mean(dim=1)
        x = self.generator(x)
        # print(x.shape)
        return x, attn
