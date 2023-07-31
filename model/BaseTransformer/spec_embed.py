import torch
import torch.nn as nn

from einops import rearrange


class SpecEmbedding(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.linear = nn.Linear(1, d_model)

    def forward(self, spec):
        batch, _ = spec.shape

        spec = rearrange(spec, "b n -> (b n) 1")
        spec_embed = self.linear(spec)
        spec_embed = rearrange(spec_embed, "(b n) d_model -> b n d_model", b=batch)

        return spec_embed


class SpecEmbedding_9to4(nn.Module):
    def __init__(self, d_model) -> None:
        super().__init__()
        self.comp = nn.Linear(1, d_model)
        self.cond = nn.Linear(2, d_model)
        self.evap = nn.Linear(3, d_model)
        self.chiller = nn.Linear(3, d_model)

    def forward(self, spec):
        comp = self.comp(rearrange(spec[:, 0], "b -> b 1")).unsqueeze(1)
        cond = self.cond(spec[:, 1:3]).unsqueeze(1)
        evap = self.evap(torch.cat((spec[:, 3:5], rearrange(spec[:, 7], "b -> b 1")), dim=1)).unsqueeze(1)
        chiller = self.chiller(torch.cat((spec[:, 5:7], rearrange(spec[:, 8], "b -> b 1")), dim=1)).unsqueeze(1)

        spec = torch.cat((comp, cond, evap, chiller), dim=1)

        return spec
