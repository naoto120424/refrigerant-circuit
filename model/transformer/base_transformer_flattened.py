import torch
from torch import nn
import math

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


class InputEmbedding(nn.Module):
    def __init__(self, num_all_features, dim):
        super(InputEmbedding, self).__init__()
        self.num_all_features = num_all_features

        self.input_emb_list = clones(nn.Linear(1, dim), self.num_all_features)

        self.positional_embedding = PositionalEmbedding(dim)

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
        input_emb_all += self.positional_embedding(input_emb_all)
        # print('input emb all', input_emb_all.shape)

        return input_emb_all


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
    def __init__(self, dim, heads=20, dim_head=1600, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout)) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, "b h n d -> b n (h d)")
        return self.to_out(out)


class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, fc_dim, dropout=0.1):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(
                nn.ModuleList(
                    [
                        PreNorm(dim, Attention(dim, heads=heads, dim_head=dim_head, dropout=dropout)),
                        PreNorm(dim, FeedForward(dim, fc_dim, dropout=dropout)),
                    ]
                )
            )

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class BaseTransformer(nn.Module):
    def __init__(self, cfg, look_back=20, dim=512, depth=3, heads=8, fc_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()

        self.num_control_features = cfg.NUM_CONTROL_FEATURES
        self.num_pred_features = cfg.NUM_PRED_FEATURES
        self.num_byproduct_features = cfg.NUM_BYPRODUCT_FEATURES
        self.num_target_features = cfg.NUM_TARGET_FEATURES
        self.num_all_features = cfg.NUM_ALL_FEATURES
        self.look_back = look_back

        self.input_embedding = InputEmbedding(self.num_all_features, dim)

        self.dropout = nn.Dropout(emb_dropout)

        self.spec_emb_list = clones(nn.Linear(1, dim), self.num_control_features)

        self.transformer = Transformer(dim, depth, heads, dim_head, fc_dim, dropout)

        self.generator = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, self.num_pred_features))

    def forward(self, input, spec):
        x = self.input_embedding(input)

        spec = torch.unsqueeze(spec, 1)  # bx9 -> bx1x9
        spec = torch.unsqueeze(spec, 1)  # bx1x9 -> bx1x1x9

        for i, spec_embedding in enumerate(self.spec_emb_list):
            if i == 0:
                spec_emb_all = spec_embedding(spec[:, :, :, i])
            else:
                spec_emb = spec_embedding(spec[:, :, :, i])
                spec_emb_all = torch.cat((spec_emb_all, spec_emb), dim=1)

        x = torch.cat((x, spec_emb_all), dim=1)

        x = self.dropout(x)
        x = self.transformer(x)
        x = x.mean(dim=1)
        x = self.generator(x)
        return x
