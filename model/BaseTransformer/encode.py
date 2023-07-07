import torch
import torch.nn as nn
import math


class AgentEncoding(nn.Module):
    def __init__(self, d_model, ts_len, max_len=5000):
        super(AgentEncoding, self).__init__()
        self.ts_len = ts_len
        ae = torch.zeros(max_len, d_model).float()
        ae.requires_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        ae[:, 0::2] = torch.sin(position * div_term)
        ae[:, 1::2] = torch.cos(position * div_term)

        ae = ae.unsqueeze(0)
        self.register_buffer("ae", ae)

    def forward(self, x):
        num_a = int(x.size(1) / self.ts_len)
        ae = self.ae[:, :num_a]
        ae = ae.repeat_interleave(self.ts_len, dim=1)

        return ae[:, : x.size(1)]


class TimeEncoding(nn.Module):
    def __init__(self, d_model, ts_len, max_len=5000):
        super(TimeEncoding, self).__init__()
        self.ts_len = ts_len
        te = torch.zeros(max_len, d_model).float()
        te.requires_grad = False

        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        te[:, 0::2] = torch.sin(position * div_term)
        te[:, 1::2] = torch.cos(position * div_term)

        te = te.unsqueeze(0)
        self.register_buffer("te", te)

    def forward(self, x):
        te = self.te[:, : self.ts_len]
        num_a = int(x.size(1) / self.ts_len)
        te = te.repeat(1, num_a, 1)

        return te[:, : x.size(1)]


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
