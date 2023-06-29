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
    MAX_EPOCH = 100
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
    torch.set_float32_matmul_precision("high")


# model decide from model name
def modelDecision(args, cfg):
    # print(vars(cfg))
    # print(args)

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
    return None
