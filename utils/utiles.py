import torch
import torch.nn as nn
import random
import os
import numpy as np

target_kW = {"ACDS_kW", "Comp_kW", "Eva_kW"}

score_list_dict = {
            "ACDS_kW": {
                'mse': [],
                'mae': [],
                'fde': []
            },
            "Comp_kW": {
                'mse': [],
                'mae': [],
                'fde': []
            },
            "Eva_kW": {
                'mse': [],
                'mae': [],
                'fde': []
            }
        }

model_list = {
    'LSTM',
    'BaseTransformer',
    'BaseTransformer_only1pe',
    'BaseTransformer_cls_pool',
    'BaseTransformer_only1pe_cls_pool',
    'BaseTransformer_agent_first'
}

criterion_list = {
    'MSE': nn.MSELoss(),
    'L1': nn.L1Loss()
}

def modelDecision(model, look_back, dim, depth, heads, fc_dim, dim_head, dropout, emb_dropout):
    if model == 'LSTM':
        from model.lstm import LSTMClassifier
        return LSTMClassifier(num_hidden_units=dim, num_layers=depth, dropout=dropout)
    elif model == 'BaseTransformer':
        from model.base_transformer import BaseTransformer
    elif model == 'BaseTransformer_only1pe':
        from model.base_transformer_only1pe import BaseTransformer
    elif model == 'BaseTransformer_cls_pool':
        from model.base_transformer_cls_pool import BaseTransformer
    elif model == 'BaseTransformer_only1pe_cls_pool':
        from model.base_transformer_only1pe_cls_pool import BaseTransformer
    elif model == 'BaseTransformer_agent_first':
        from model.base_transformer_agent_first import BaseTransformer

    return BaseTransformer(look_back=look_back, dim=dim, depth=depth, heads=heads, fc_dim=fc_dim,
                           dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)

def deviceChecker():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
