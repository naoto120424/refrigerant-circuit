import torch
import torch.nn as nn
import random
import os
import numpy as np
from model.lstm import LSTMClassifier
from model.base_transformer import BaseTransformer

target_kW = {"ACDS_kW", "Comp_kW", "Eva_kW"}

model_list = {
    'lstm': LSTMClassifier() # ,
    # 'BaseTransformer': BaseTransformer()
}

criterion_list = {
    'mse': nn.MSELoss(),
    'l1': nn.L1Loss()
}

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
