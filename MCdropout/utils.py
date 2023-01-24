import torch
import torch.nn as nn
import numpy as np
import random, os


class CFG:
    NUM_FIXED_DATA = 8
    NUM_CONTROL_FEATURES = 6
    NUM_PRED_FEATURES = 30
    NUM_BYPRODUCT_FEATURES = 27
    NUM_TARGET_FEATURES = 3
    NUM_ALL_FEATURES = 36
    RESULT_PATH = os.path.join("..", "..", "result")
    DATA_PATH = os.path.join("..", "..", "dataset")


predict_time_list = []

fixed_case_list = {
    "fixed_case0001_Comp_rpm_Swing_test_VF_Fix",
    "fixed_case0002_Comp_rpm_Swing_test_VF_Fix",
    "fixed_case0003_Comp_rpm_Swing_test_VF_Fix",
    "fixed_case0004_Comp_rpm_Swing_test_VF_Fix",
    "fixed_case0001_EXV_Swing_test_VF_Fix",
    "fixed_case0002_EXV_Swing_test_VF_Fix",
    "fixed_case0003_EXV_Swing_test_VF_Fix",
    "fixed_case0004_EXV_Swing_test_VF_Fix",
}

model_list = {
    "LSTM",
    "BaseTransformer",
    "BaseTransformer_sensor_first",
    "BaseTransformer_3types_aete",
    "BaseTransformer_3types_AgentAwareAttention",
    "BaseTransformer_individually_aete",
    "BaseTransformer_individually_AgentAwareAttention",
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

        return BaseTransformer(cfg, args.look_back, args.dim, args.depth, args.heads, args.fc_dim, args.dim_head, args.dropout, args.emb_dropout)

    return None


# case name decide from test index for step1 test folder
def CaseNameDecision(test_index):
    if test_index not in np.arange(CFG.NUM_FIXED_DATA):
        case_name = f"case{str(test_index-CFG.NUM_FIXED_DATA+1).zfill(4)}"
    else:
        case_name = list(fixed_case_list)[test_index]

    return case_name


# enable the dropout layers during test-time
def enable_dropout(model):
    for m in model.modules():
        if m.__class__.__name__.startswith("Dropout"):
            m.train()
