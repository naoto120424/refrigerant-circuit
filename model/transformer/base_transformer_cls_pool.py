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
        div_term = (
            torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)
        ).exp()

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


class Attention(nn.Module):
    def __init__(self, dim, heads=20, dim_head=1600, dropout=0.1):
        super().__init__()
        inner_dim = dim_head * heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head**-0.5

        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias=False)

        self.to_out = (
            nn.Sequential(nn.Linear(inner_dim, dim), nn.Dropout(dropout))
            if project_out
            else nn.Identity()
        )

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
                        PreNorm(
                            dim,
                            Attention(
                                dim, heads=heads, dim_head=dim_head, dropout=dropout
                            ),
                        ),
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
    def __init__(
        self,
        look_back,
        dim=512,
        depth=3,
        heads=8,
        fc_dim=2048,
        dim_head=64,
        dropout=0.1,
        emb_dropout=0.1,
    ):
        super().__init__()

        self.num_all_features = 39
        self.num_control_features = 9
        self.num_pred_features = 30
        self.num_byproduct_features = 27
        self.num_target_features = 3
        self.look_back = look_back

        self.input_embedding = nn.Linear(self.num_all_features, dim)
        self.positional_embedding = PositionalEmbedding(dim)  # 絶対位置エンコーディング

        self.pos_embedding = nn.Parameter(
            torch.randn(
                1,
                self.num_pred_features + self.look_back + self.num_control_features,
                dim,
            )
        )
        self.pred_token = nn.Parameter(torch.randn(1, self.num_pred_features, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.spec_emb_list = clones(nn.Linear(1, dim), self.num_control_features)

        self.transformer = Transformer(dim, depth, heads, dim_head, fc_dim, dropout)

        self.generator = nn.Sequential(
            nn.Linear(self.num_pred_features * dim, dim),
            nn.ReLU(inplace=True),
            nn.LayerNorm(dim),
            nn.Linear(dim, self.num_pred_features),
        )

    def forward(self, input, spec):
        x = self.input_embedding(input)
        x += self.positional_embedding(x)

        b, _, _ = x.shape

        pred_tokens = repeat(self.pred_token, "() n d -> b n d", b=b)
        x = torch.cat((pred_tokens, x), dim=1)

        spec = torch.unsqueeze(spec, 1)  # bx9 -> bx1x9
        spec = torch.unsqueeze(spec, 1)  # bx1x9 -> bx1x1x9

        for i, spec_embedding in enumerate(self.spec_emb_list):
            if i == 0:
                spec_emb_all = spec_embedding(spec[:, :, :, i])
            else:
                spec_emb = spec_embedding(spec[:, :, :, i])
                spec_emb_all = torch.cat((spec_emb_all, spec_emb), dim=1)

        x = torch.cat((x, spec_emb_all), dim=1)

        x += self.pos_embedding
        x = self.dropout(x)
        x = self.transformer(x)
        # print(x.shape)
        x = x[:, : self.num_pred_features]
        x = rearrange(x, "b pred_token dim -> b (pred_token dim)")
        # x = x.mean(dim=1)
        # print('x.shape', x.shape)
        x = self.generator(x)
        return x
