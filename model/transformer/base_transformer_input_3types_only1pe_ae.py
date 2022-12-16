import torch
from torch import nn
import math
import matplotlib.pyplot as plt

from copy import deepcopy
from einops import rearrange, repeat


def pair(t):
    return t if isinstance(t, tuple) else (t, t)


def clones(module, n):
    # produce N identical layers.
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


# classes
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_len=5000):
        super(PositionalEncoding, self).__init__()
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


class AgentEncoding(nn.Module):
    def __init__(self, d_model, look_back, num_control_features, max_len=5000):
        super(AgentEncoding, self).__init__()
        self.d_model = d_model
        self.look_back = look_back
        self.num_control_features = num_control_features
        ae = torch.zeros(max_len, d_model).float()
        ae.requires_grad = False
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()
        ae[:, 0::2] = torch.sin(position * div_term)
        ae[:, 1::2] = torch.cos(position * div_term)
        ae = ae.unsqueeze(0)
        self.register_buffer("ae", ae)

    def forward(self, x):
        # print(x.shape)
        ae = self.ae[:, : self.look_back]
        # ae_control_features = self.ae[:, self.look_back : self.look_back + self.num_control_features]
        # num_a = int((x.size(1) - self.num_control_features) / self.look_back)
        num_a = int(x.size(1) / self.look_back)
        # print(num_a)
        ae = ae.repeat(1, num_a, 1)
        # ae = torch.cat([ae, ae_control_features], dim=1)
        # print(ae.shape)
        return ae[:, : x.size(1)]


class InputEmbedding(nn.Module):
    def __init__(self, look_back, num_control_features, num_byproduct_features, num_target_features, dim):
        super(InputEmbedding, self).__init__()
        self.look_back = look_back
        self.num_control_features = num_control_features
        self.num_byproduct_features = num_byproduct_features
        self.num_target_features = num_target_features

        self.control_embedding = nn.Linear(num_control_features, dim)
        self.byproduct_embedding = nn.Linear(num_byproduct_features, dim)
        self.target_embedding = nn.Linear(num_target_features, dim)

        self.positional_encoding = PositionalEncoding(dim)
        self.agent_encoding = AgentEncoding(dim, self.look_back, self.num_control_features)

    def forward(self, x):
        control = self.control_embedding(x[:, :, : self.num_control_features])
        # control += self.positional_embedding(control)
        # print('control embedding: ', control.shape)
        byproduct = self.byproduct_embedding(x[:, :, self.num_control_features : self.num_control_features + self.num_byproduct_features])
        # byproduct += self.positional_embedding(byproduct)
        # print('byproduct embedding: ', byproduct.shape)
        target = self.target_embedding(x[:, :, self.num_control_features + self.num_byproduct_features :])
        # target += self.positional_embedding(target)
        # print('target embedding: ', target.shape)

        x = torch.cat([control, byproduct, target], dim=1)
        x += self.positional_encoding(x)
        x += self.agent_encoding(x)

        """
        # positional encoding visualization
        pe = self.positional_encoding(x).to("cpu").detach().numpy().copy()
        print("pe control", pe.shape)
        fig = plt.figure()
        plt.imshow(pe[0])
        plt.colorbar()
        plt.savefig("img/positional_encoding_input_3types.png")
        # agent encoding visualization
        ae = self.agent_encoding(x).to("cpu").detach().numpy().copy()
        print("ae", ae.shape)
        fig = plt.figure()
        plt.imshow(ae[0])
        plt.colorbar()
        plt.savefig("img/agent_encoding_input_3types.png")
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
    def __init__(self, look_back=20, dim=512, depth=3, heads=8, fc_dim=2048, dim_head=64, dropout=0.1, emb_dropout=0.1):
        super().__init__()

        self.num_all_features = 36
        self.num_control_features = 6
        self.num_pred_features = 30
        self.num_byproduct_features = 27
        self.num_target_features = 3
        self.look_back = look_back

        self.input_embedding = InputEmbedding(self.look_back, self.num_control_features, self.num_byproduct_features, self.num_target_features, dim)

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
