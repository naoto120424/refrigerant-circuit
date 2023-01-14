import torch
import torch.nn as nn
import numpy as np
import random, os


num_fixed_data = 8
num_control_features = 6
num_all_features = 36

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


# model Decide function from model name
def modelDecision(args):
    if "LSTM" in args.model:
        if args.model == "LSTM":
            from model.lstm.lstm import LSTMClassifier

        return LSTMClassifier(args.look_back, args.depth, args.dim, args.dropout)

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

        return BaseTransformer(args.look_back, args.dim, args.depth, args.heads, args.fc_dim, args.dim_head, args.dropout, args.emb_dropout)

    return None


def CaseNameDecision(test_index):
    if test_index not in np.arange(num_fixed_data):
        case_name = f"case{str(test_index-num_fixed_data+1).zfill(4)}"
    else:
        case_name = list(fixed_case_list)[test_index]

    return case_name
