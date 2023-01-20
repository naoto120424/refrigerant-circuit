import torch
import torch.nn as nn
import numpy as np
import random, os


class CFG:
    NUM_FIXED_DATA = 8
    NUM_CONTROL_FEATURES = 9
    NUM_PRED_FEATURES = 41
    NUM_BYPRODUCT_FEATURES = 37
    NUM_TARGET_FEATURES = 4
    NUM_ALL_FEATURES = 50
    RESULT_PATH = os.path.join("..", "result")
    DATA_PATH = os.path.join("..", "teacher")


predict_time_list = []

model_list = {
    "LSTM",
    "BaseTransformer",
    "BaseTransformer_sensor_first",
    "BaseTransformer_3types_aete",
    "BaseTransformer_3types_AgentAwareAttention",
    "BaseTransformer_flattened_aete",
    "BaseTransformer_flattened_AgentAwareAttention",
}

criterion_list = {"MSE": nn.MSELoss(), "L1": nn.L1Loss()}


# seed
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# model decide from model name
def modelDecision(args, cfg):
    if "LSTM" in args.model:
        if args.model == "LSTM":
            from model.lstm.lstm import LSTMClassifier

        return LSTMClassifier(cfg, args.look_back, args.depth, args.dim, args.dropout)

    if "BaseTransformer" in args.model:
        if args.model == "BaseTransformer":
            from model.transformer.base_transformer import BaseTransformer
        elif args.model == "BaseTransformer_sensor_first":
            from model.transformer.base_transformer_sensor_first import BaseTransformer
        elif args.model == "BaseTransformer_3types_aete":
            from model.transformer.base_transformer_3types_aete import BaseTransformer
        elif args.model == "BaseTransformer_3types_AgentAwareAttention":
            from model.transformer.base_transformer_3types_AgentAwareAttention import BaseTransformer
        elif args.model == "BaseTransformer_flattened_aete":
            from model.transformer.base_transformer_flattened_aete import BaseTransformer
        elif args.model == "BaseTransformer_flattened_AgentAwareAttention":
            from model.transformer.base_transformer_flattened_AgentAwareAttention import BaseTransformer

        return BaseTransformer(cfg, args.look_back, args.dim, args.depth, args.heads, args.fc_dim, args.dim_head, args.dropout, args.emb_dropout)

    return None
