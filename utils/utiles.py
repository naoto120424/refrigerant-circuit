import torch
import torch.nn as nn
import random
import os
import numpy as np


target_kW = {"ACDS_kW", "Comp_kW", "Eva_kW"}

score_list_dict = {
    "ACDS_kW": {"ade": [], "fde": []},
    "Comp_kW": {"ade": [], "fde": []},
    "Eva_kW": {"ade": [], "fde": []},
}

fixed_case_list = {
    "case0001_Comp_rpm_Swing_test_VF_Fix",
    "case0002_Comp_rpm_Swing_test_VF_Fix",
    "case0003_Comp_rpm_Swing_test_VF_Fix",
    "case0004_Comp_rpm_Swing_test_VF_Fix",
    "case0001_EXV_Swinbg_test_VF_Fix",
    "case0002_EXV_Swinbg_test_VF_Fix",
    "case0003_EXV_Swinbg_test_VF_Fix",
    "case0004_EXV_Swinbg_test_VF_Fix",
}

model_list = {
    "LSTM",
    "LSTM_input_sensor_first",
    "LSTM_input_3types",
    "LSTM_input_flattened",
    "LSTM_input_individually",
    "BaseTransformer",
    "BaseTransformer_only1pe",
    "BaseTransformer_only1pe_ae",
    "BaseTransformer_input_sensor_first",
    "BaseTransformer_input_sensor_first_only1pe",
    "BaseTransformer_input_3types",
    "BaseTransformer_input_3types_only1pe",
    "BaseTransformer_input_flattened",
    "BaseTransformer_input_flattened_only1pe",
}

criterion_list = {"MSE": nn.MSELoss(), "L1": nn.L1Loss()}


def modelDecision(model, look_back, dim, depth, heads, fc_dim, dim_head, dropout, emb_dropout):
    if "LSTM" in model:
        if model == "LSTM":
            from model.lstm.lstm import LSTMClassifier
        elif model == "LSTM_input_sensor_first":
            from model.lstm.lstm_input_sensor_first import LSTMClassifier
        elif model == "LSTM_input_3types":
            from model.lstm.lstm_input_3types import LSTMClassifier
        elif model == "LSTM_input_flattened":
            from model.lstm.lstm_input_flattened import LSTMClassifier
        elif model == "LSTM_input_individually":
            from model.lstm.lstm_input_individually import LSTMClassifier

        return LSTMClassifier(look_back=look_back, num_hidden_units=dim, num_layers=depth, dropout=dropout)

    if "BaseTransformer" in model:
        if model == "BaseTransformer":
            from model.transformer.base_transformer import BaseTransformer
        elif model == "BaseTransformer_only1pe":
            from model.transformer.base_transformer_only1pe import BaseTransformer
        elif model == "BaseTransformer_only1pe_ae":
            from model.transformer.base_transformer_only1pe_ae import BaseTransformer
        elif model == "BaseTransformer_input_sensor_first":
            from model.transformer.base_transformer_input_sensor_first import BaseTransformer
        elif model == "BaseTransformer_input_sensor_first_only1pe":
            from model.transformer.base_transformer_input_sensor_first_only1pe import BaseTransformer
        elif model == "BaseTransformer_input_3types":
            from model.transformer.base_transformer_input_3types import BaseTransformer
        elif model == "BaseTransformer_input_3types_only1pe":
            from model.transformer.base_transformer_input_3types_only1pe import BaseTransformer
        elif model == "BaseTransformer_input_flattened":
            from model.transformer.base_transformer_input_flattened import BaseTransformer
        elif model == "BaseTransformer_input_flattened_only1pe":
            from model.transformer.base_transformer_input_flattened_only1pe import BaseTransformer

        return BaseTransformer(look_back=look_back, dim=dim, depth=depth, heads=heads, fc_dim=fc_dim, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)

    return


def deviceChecker():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
