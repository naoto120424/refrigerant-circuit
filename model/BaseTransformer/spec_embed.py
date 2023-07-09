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
