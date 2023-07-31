import torch
import torch.nn as nn
import numpy as np
import random, os


class CFG:
    NUM_CONTROL_FEATURES = 9
    NUM_PRED_FEATURES = 41
    NUM_BYPRODUCT_FEATURES = 37
    NUM_TARGET_FEATURES = 4
    NUM_ALL_FEATURES = 50
    DATA_PATH = os.path.join("..", "teacher")
    RESULT_PATH = os.path.join("..", "result")
    MLFLOW_PATH = os.path.join("..", "mlflow_experiment")


model_list = {
    "LSTM",
    "BaseTransformer",
    "BaseTransformer_sensor_first",
    "BaseTransformer_3types_aete",
    "BaseTransformer_3types_AgentAwareAttention",
    "BaseTransformer_individually_aete",
    "BaseTransformer_individually_AgentAwareAttention",
    "Transformer",
    "Crossformer",
}

criterion_list = {"MSE": nn.MSELoss(), "L1": nn.L1Loss()}

predict_time_list = []


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

        return LSTMClassifier(cfg, args)

    if "BaseTransformer" in args.model:
        if args.model == "BaseTransformer":
            from model.BaseTransformer.base_transformer import BaseTransformer
        elif args.model == "BaseTransformer_sensor_first":
            from model.BaseTransformer.base_transformer_sensor_first import BaseTransformer
        elif args.model == "BaseTransformer_3types_aete":
            from model.BaseTransformer.base_transformer_3types_aete import BaseTransformer
        elif args.model == "BaseTransformer_3types_AgentAwareAttention":
            from model.BaseTransformer.base_transformer_3types_AgentAwareAttention import BaseTransformer
        elif args.model == "BaseTransformer_individually_aete":
            from model.BaseTransformer.base_transformer_individually_aete import BaseTransformer
        elif args.model == "BaseTransformer_individually_AgentAwareAttention":
            from model.BaseTransformer.base_transformer_individually_AgentAwareAttention import BaseTransformer

        return BaseTransformer(cfg, args)

    if "Transformer" in args.model:
        if args.model == "Transformer":
            from model.Transformer.transformer import Transformer

        return Transformer(cfg, args)

    if "Crossformer" in args.model:
        if args.model == "Crossformer":
            from model.crossformer.cross_former import Crossformer

        return Crossformer(cfg, args)
    return None


# change look_back -> in_len
# change dim -> d_model
# change depth -> e_layers
# change heads -> n_heads
# del dim_head: dim_head = d_model // n_heads
# change fc_dim -> d_ff
# del emb_dropout, change emb_dropout -> dropout
