import torch
import torch.nn as nn

from copy import deepcopy


def clones(module, n):
    # produce N identical layers.
    assert isinstance(module, nn.Module)
    return nn.ModuleList([deepcopy(module) for _ in range(n)])


class LSTMClassifier(nn.Module):
    def __init__(self, look_back, num_layers=2, num_hidden_units=256, spec_dim=6, output_dim=30, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.input_dim = 36
        self.num_hidden_units = num_hidden_units
        self.lstm_list = clones(
            nn.LSTM(
                input_size=1,
                hidden_size=num_hidden_units,
                num_layers=num_layers,
                dropout=dropout,
                batch_first=True,
            ),
            self.input_dim,
        )
        self.spec_dense = nn.Linear(spec_dim, num_hidden_units)
        self.predicter = nn.Sequential(
            nn.Linear(num_hidden_units * (self.input_dim + 1), num_hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden_units, output_dim),
        )

    def forward(self, x, spec, h=None):
        x = torch.unsqueeze(x, 2)
        # print(x.shape)
        for i, lstm in enumerate(self.lstm_list):
            if i == 0:
                hidden1_all, _ = lstm(x[:, :, :, i], h)
                hidden1_all = hidden1_all[:, -1, :]
                # print(hidden1_all.shape)
            else:
                hidden1, _ = lstm(x[:, :, :, i], h)
                hidden1 = hidden1[:, -1, :]
                hidden1_all = torch.cat([hidden1_all, hidden1], dim=1)
        # print(hidden1_all.shape)
        hidden2 = self.spec_dense(spec)
        # print(hidden2.shape)
        y = self.predicter(torch.cat([hidden1_all, hidden2], dim=1))  # 最後のセルだけを取り出している
        return y
