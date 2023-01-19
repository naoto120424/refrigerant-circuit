import torch
import torch.nn as nn


class LSTMClassifier(nn.Module):
    def __init__(self, cfg, look_back, num_layers=2, num_hidden_units=256, dropout=0.2):
        super(LSTMClassifier, self).__init__()
        self.num_control_features = cfg.NUM_CONTROL_FEATURES
        self.num_pred_features = cfg.NUM_PRED_FEATURES
        self.num_byproduct_features = cfg.NUM_BYPRODUCT_FEATURES
        self.num_target_features = cfg.NUM_TARGET_FEATURES
        self.num_all_features = cfg.NUM_ALL_FEATURES
        self.look_back = look_back

        self.lstm_control = nn.LSTM(
            input_size=self.num_control_features,
            hidden_size=num_hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.lstm_byproduct = nn.LSTM(
            input_size=self.num_byproduct_features,
            hidden_size=num_hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.lstm_target = nn.LSTM(
            input_size=self.num_target_features,
            hidden_size=num_hidden_units,
            num_layers=num_layers,
            dropout=dropout,
            batch_first=True,
        )

        self.spec_dense = nn.Linear(self.num_all_features, num_hidden_units)
        self.predictor = nn.Sequential(
            nn.Linear(num_hidden_units * 4, num_hidden_units),
            nn.ReLU(inplace=True),
            nn.Linear(num_hidden_units, self.num_pred_features),
        )

    def forward(self, x, spec, h=None):
        hidden1, _ = self.lstm_control(x[:, :, : self.num_control_features], h)
        hidden2, _ = self.lstm_byproduct(x[:, :, self.num_control_features : self.num_control_features + self.num_byproduct_features], h)
        hidden3, _ = self.lstm_target(x[:, :, self.num_control_features + self.num_byproduct_features :], h)
        hidden4 = self.spec_dense(spec)
        y = self.predictor(torch.cat([hidden1[:, -1, :], hidden2[:, -1, :], hidden3[:, -1, :], hidden4], dim=1))  # 最後のセルだけを取り出している
        return y
