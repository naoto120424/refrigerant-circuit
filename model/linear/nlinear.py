import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from einops import rearrange


class Model(nn.Module):
    """
    Normalization-Linear
    """

    def __init__(self, cfg, args):
        super(Model, self).__init__()
        self.seq_len = args.in_len
        self.pred_len = args.out_len

        # Use this line if you want to visualize the weights
        # self.Linear.weight = nn.Parameter((1/self.seq_len)*torch.ones([self.pred_len,self.seq_len]))
        self.channels = cfg.NUM_ALL_FEATURES
        self.individual = True
        if self.individual:
            self.Linear = nn.ModuleList()
            for i in range(self.channels):
                self.Linear.append(nn.Linear(self.seq_len, self.pred_len))
        else:
            self.Linear = nn.Linear(self.seq_len, self.pred_len)

    def forward(self, x, spec):
        # x: [Batch, Input length, Channel]
        seq_last = x[:, -1:, :].detach()
        x = x - seq_last
        if self.individual:
            output = torch.zeros([x.size(0), self.pred_len, x.size(2)], dtype=x.dtype).to(x.device)
            for i in range(self.channels):
                output[:, :, i] = self.Linear[i](x[:, :, i])
            x = output
        else:
            x = self.Linear(x.permute(0, 2, 1)).permute(0, 2, 1)
        # print(x.shape, seq_last.shape)
        x = x + seq_last
        # print(x.shape)
        return rearrange(x[:, :, 9:], "b 1 x -> b x"), None  # [Batch, Output length, Channel]
