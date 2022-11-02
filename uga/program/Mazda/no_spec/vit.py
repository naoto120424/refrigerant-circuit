# Mazda vit

import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

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
    def __init__(self, num_pred, patch_split, dim, depth, heads, dim_head, fc_dim = 2048, dropout = 0.1, emb_dropout = 0.1):
        super().__init__()
        
        x_dim = 40
        y_dim = 40
        z_dim = 20
        data_dim = 12
    
        patch_x = x_dim // patch_split[0] 
        patch_y = y_dim // patch_split[1]
        patch_z = z_dim // patch_split[2]

        num_patches = (x_dim // patch_x) * (y_dim // patch_y) * (z_dim // patch_z)

        patch_dim = patch_x * patch_y * patch_z * data_dim
        
        self.to_patch_embedding = nn.Sequential(
            Rearrange('b (x p1) (y p2) (z p3) data -> b (x y z) (p1 p2 p3 data)', p1=patch_x, p2=patch_y, p3=patch_z),
            nn.Linear(patch_dim, dim),
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
        x = self.to_patch_embedding(data)
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