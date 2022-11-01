from operator import truediv
import torch
import random
import os
import numpy as np
from model.lstm import *


class CFG:
    EPOCH = 200
    LR = 1e-4
    WEIGHT_DECAY = 1e-3
    DATA_DIR = '../mazda/dataset/'
    FOLD_NUM = 5

def make_experiment_id_and_path():
    experiment_path = '../result'
    dir_list = os.listdir(experiment_path)
    prev_experiment_id = max([int(x) for x in dir_list])
    current_experiment_id = prev_experiment_id + 1
    current_experiment_path = os.path.join(experiment_path, str(current_experiment_id))
    os.makedirs(current_experiment_path, exist_ok=True)
    os.makedirs(os.path.join(current_experiment_path, 'csv'), exist_ok=True)
    return current_experiment_id, current_experiment_path
    

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
