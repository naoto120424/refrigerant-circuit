import torch
import random
import os
import numpy as np
from model.lstm import *


class CFG:
    EPOCH = 200
    LR = 1e-4
    WEIGHT_DECAY = 1e-3
    DATA_DIR = '../dataset/'
    FOLD_NUM = 5
    

def deviceChecker():
    if torch.cuda.is_available():
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
