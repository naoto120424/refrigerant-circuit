import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, cfg, args):
        super(LSTMClassifier, self).__init__()
        self.input_dim = cfg.NUM_ALL_FEATURES
        self.spec_dim = cfg.NUM_CONTROL_FEATURES
        self.output_dim = cfg.NUM_PRED_FEATURES
        self.num_hidden_units = args.d_model
        self.num_layers = args.e_layers
        self.dropout = args.dropout

        self.lstm = nn.LSTM(
            input_size=self.input_dim,
            hidden_size=self.num_hidden_units,
            num_layers=self.num_layers,
            dropout=self.dropout,
            batch_first=True,
        )
        self.spec_dense = nn.Linear(self.spec_dim, self.num_hidden_units)
        self.predictor = nn.Sequential(
            nn.Linear(self.num_hidden_units * 2, self.num_hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(self.num_hidden_units, self.output_dim),
        )

    def forward(self, x, spec, h=None):
        hidden1, _ = self.lstm(x, h)
        hidden2 = self.spec_dense(spec)
        y = self.predictor(torch.cat([hidden1[:, -1, :], hidden2], dim=1))  # 最後のセルだけを取り出している
        return y, _
