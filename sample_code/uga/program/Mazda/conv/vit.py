# Mazda vit

import torch
from torch import nn

from copy import deepcopy
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

def clones(module, n):
    """
    Produce N identical layers.
    """
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])

# classes

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv_block = nn.Sequential(
            nn.Conv3d(in_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=True),
            nn.ReLU(inplace=True),
            nn.MaxPool3d(kernel_size=2),
        )

    def forward(self, x):
        return self.conv_block(x)

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )
    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 20, dim_head = 1600, dropout = 0.1):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, fc_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, fc_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x

class ViT(nn.Module):
    def __init__(self, num_pred, dim, depth, heads, dim_head, fc_dim = 2048, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        
        x_dim = 40
        y_dim = 40
        z_dim = 20
        data_dim = 12

        self.conv1 = ConvBlock(12, 12)
        self.conv2 = ConvBlock(12, 12)

        num_patches = x_dim // 4 * y_dim // 4 * z_dim // 4 # convの深さによって変える
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b x y z data -> b (x y z) data'),
            nn.Linear(data_dim, dim),
        )
        
        self.num_pred = num_pred

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + num_pred, dim))
        self.pred_token = nn.Parameter(torch.randn(1, num_pred, dim))
        self.dropout = nn.Dropout(emb_dropout)

        dim_head = num_patches
        self.transformer = Transformer(dim, depth, heads, dim_head, fc_dim, dropout)

        self.to_latent = nn.Identity()

        self.generator = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_pred),
            nn.ReLU(inplace=True)
        )

    def forward(self, data):
        #print("input:", data.shape)
        x = data.permute(0, 4, 1, 2, 3) # dataの次元の入れ替え bx12x40x40x20

        x = self.conv1(x) # bx12x20x20x10
        x = self.conv2(x) # bx12x10x10x5

        x = x.permute(0, 2, 3, 4, 1) # bx10x10x5x12

        x = self.to_patch_embedding(x)
        #print("patch_emb:", x.shape)
        b, n, _ = x.shape

        pred_tokens = repeat(self.pred_token, '() n d -> b n d', b = b)
        x = torch.cat((pred_tokens, x), dim=1)

        x += self.pos_embedding[:, :(n + self.num_pred)]
        #print("pos_emb:", x.shape)
        x = self.dropout(x)
        #print("dropout:", x.shape)

        x = self.transformer(x)
        #print("out_TF:", x.shape)
        x = x.mean(dim = 1)   
        #print("pool_TF:", x.shape)  
        x = self.to_latent(x)
        
        #print("out:", self.generator(x).shape)

        x = self.generator(x)

        return x