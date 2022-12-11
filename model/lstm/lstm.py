import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, look_back, num_layers=2, num_hidden_units=256, spec_dim=6, output_dim=30, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.input_dim = 36
        self.num_hidden_units = num_hidden_units
        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=num_hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.spec_dense = nn.Linear(spec_dim, num_hidden_units)
        self.predicter = nn.Sequential(
            nn.Linear(num_hidden_units * 2, num_hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden_units, output_dim),
        )

    def forward(self, x, spec, h=None):
        hidden1, _ = self.lstm(x, h)
        hidden2 = self.spec_dense(spec)
        y = self.predicter(torch.cat([hidden1[:, -1, :], hidden2], dim=1))  # 最後のセルだけを取り出している
        return y
