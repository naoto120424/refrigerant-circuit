import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, look_back=20, num_layers=2, num_hidden_units=256, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.input_dim = 36
        self.spec_dim = 6
        self.output_dim = 30
        self.num_hidden_units = num_hidden_units
        self.lstm = nn.LSTM(
            input_size=look_back,
            hidden_size=num_hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )
        self.spec_dense = nn.Linear(self.spec_dim, num_hidden_units)
        self.predicter = nn.Sequential(
            nn.Linear(num_hidden_units * 2, num_hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden_units, self.output_dim),
        )

    def forward(self, x, spec, h=None):
        x = torch.permute(x, (0, 2, 1))
        hidden1, _ = self.lstm(x, h)
        hidden2 = self.spec_dense(spec)
        y = self.predicter(torch.cat([hidden1[:, -1, :], hidden2], dim=1))  # 最後のセルだけを取り出している
        return y
