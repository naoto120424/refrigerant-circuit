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
        self.look_back = args.look_back
        self.num_all_features = cfg.NUM_ALL_FEATURES
        self.num_control_features = cfg.NUM_CONTROL_FEATURES

        self.input_emb_list = clones(nn.Linear(1, args.dim), self.num_all_features)

        self.time_encoding = TimeEncoding(args.dim, self.look_back)
        self.agent_encoding = AgentEncoding(args.dim, self.look_back)

    def forward(self, x):
        x = torch.unsqueeze(x, 2)
        # print(x.shape)
        for i, input_embedding in enumerate(self.input_emb_list):
            if i == 0:
                input_emb_all = input_embedding(x[:, :, :, i])
            else:
                input_emb = input_embedding(x[:, :, :, i])
                input_emb_all = torch.cat((input_emb_all, input_emb), dim=1)
        # print(input_emb_all.shape)
        input_emb_all += self.agent_encoding(input_emb_all)
        input_emb_all += self.time_encoding(input_emb_all)
        # print('input emb all', input_emb_all.shape)

        """
        img_path = os.path.join("img", "inp_individually", "encoding")
        os.makedirs(img_path, exist_ok=True)
        # positional encoding visualization
        te = self.time_encoding(input_emb_all).to("cpu").detach().numpy().copy()
        print("te", te.shape)
        fig = plt.figure()
        plt.imshow(te[0])
        plt.colorbar()
        plt.savefig(f"img/inp_individually/encoding/time_encoding_input_individually_lookback{self.look_back}.png")

        # agent encoding visualization
        ae = self.agent_encoding(input_emb_all).to("cpu").detach().numpy().copy()
        print("ae", ae.shape)
        fig = plt.figure()
        plt.imshow(ae[0])
        plt.colorbar()
        plt.savefig(f"img/inp_individually/encoding/agent_encoding_input_individually_lookback{self.look_back}.png")
        """

        return input_emb_all


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
        img_path = os.path.join("img", "inp_individually", "attention")
        os.makedirs(img_path, exist_ok=True)
        fig = plt.figure()
        plt.imshow(attn_mask, cmap="Blues")
        plt.colorbar()
        plt.savefig(f"img/inp_individually/attention/attention_mask_input_individually_lookback{self.look_back}.png")
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
        self.num_agent = self.num_all_features

        self.input_embedding = InputEmbedding(cfg, args)
        self.spec_embedding = SpecEmbedding(args.dim, self.num_control_features)

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
        return x, attn
