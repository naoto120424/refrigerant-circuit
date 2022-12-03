import torch
import torch.nn as nn
import random
import os
import numpy as np
from model.lstm import LSTMClassifier

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
    'LSTM': LSTMClassifier()
}

transformer_list = {
    'BaseTransformer', 
    'BaseTransformer_agent_first'
}

criterion_list = {
    'MSE': nn.MSELoss(),
    'L1': nn.L1Loss()
}

def modelTransformer(model, look_back):
    if model == 'BaseTransformer':
        from model.base_transformer import BaseTransformer
    elif model == 'BaseTransformer_agent_first':
        from model.base_transformer_agent_first import BaseTransformer

    return BaseTransformer(look_back=look_back)

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
