import torch
from torch import nn


class DeepOLSTM(nn.Module):
    def __init__(self, cfg, args):
        super().__init__()

        self.num_control_features = cfg.NUM_CONTROL_FEATURES
        self.num_pred_features = cfg.NUM_PRED_FEATURES
        self.num_byproduct_features = cfg.NUM_BYPRODUCT_FEATURES
        self.num_target_features = cfg.NUM_TARGET_FEATURES
        self.num_all_features = cfg.NUM_ALL_FEATURES
        self.in_len = args.in_len
        
        self.branch1 = nn.Linear(2*args.d_model, self.num_pred_features)
        self.branch2 = nn.Linear(1, args.d_model)
        
        self.trunk = nn.ModuleList([])
        self.trunk.append(
            nn.ModuleList(
                [
                    nn.Linear(1, args.trunk_d_model),
                    nn.ReLU()
                ]
            )
        )
        for _ in range(args.trunk_layers-2):
            self.trunk.append(
                nn.ModuleList(
                    [
                        nn.Linear(args.trunk_d_model, args.trunk_d_model),
                        nn.ReLU(),
                    ]
                )
            )
        self.trunk.append(
            nn.ModuleList(
                [
                    nn.Linear(args.trunk_d_model, args.d_model),
                    nn.ReLU(),
                ]
            )
        )
        
        self.lstm = nn.LSTM(
            input_size=cfg.NUM_ALL_FEATURES,
            hidden_size=args.d_model,
            num_layers=args.e_layers,
            dropout=args.dropout,
            batch_first=True,
        )
        self.spec_dense = nn.Linear(cfg.NUM_CONTROL_FEATURES, args.d_model)

    def forward(self, input, spec, timedata, h=None):
        hidden1, _ = self.lstm(input, h)
        hidden2 = self.spec_dense(spec)
        
        x = torch.cat([hidden1[:, -1, :], hidden2], dim=1).unsqueeze(1)
        # print(x.shape)
        x = self.branch1(x)
        # print(x.shape)
        x = self.branch2(x.permute(0, 2, 1))
        # print(x.shape)
        
        y = timedata.repeat(1, self.num_pred_features).unsqueeze(2)
        
        for linear, relu in self.trunk:
            y = linear(y)
            y = relu(y)
        # print(y.shape)
        
        x = torch.sum(x * y, dim=-1, keepdim=True)
        x = x.squeeze(2)
        
        return x, None
