import torch
import torch.nn as nn

from einops import rearrange


class SpecEmbedding(nn.Module):
    def __init__(self, d_model, num_control_features=9) -> None:
        super().__init__()
        self.linear = nn.Linear(1, d_model)

    def forward(self, spec):
        batch, num_control_features = spec.shape

        spec = rearrange(spec, "b n -> (b n) 1")
        spec_embed = self.linear(spec)
        spec_embed = rearrange(spec_embed, "(b n) d_model -> b n d_model", b=batch)

        return spec_embed


"""
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
"""
