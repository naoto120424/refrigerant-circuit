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
    "fixed_case0001_EXV_Swinbg_test_VF_Fix",
    "fixed_case0002_EXV_Swinbg_test_VF_Fix",
    "fixed_case0003_EXV_Swinbg_test_VF_Fix",
    "fixed_case0004_EXV_Swinbg_test_VF_Fix",
}

model_list = {
    "LSTM",
    "LSTM_sensor_first",
    "LSTM_3types",
    "LSTM_flattened",
    "LSTM_individually",
    "BaseTransformer",
    "BaseTransformer_only1pe",
    "BaseTransformer_aete",
    "BaseTransformer_sensor_first",
    "BaseTransformer_sensor_first_only1pe",
    "BaseTransformer_sensor_first_no_pe",
    "BaseTransformer_3types",
    "BaseTransformer_3types_only1pe",
    "BaseTransformer_3types_ae",
    "BaseTransformer_3types_aete",
    "BaseTransformer_3types_AgentAwareAttention",
    "BaseTransformer_flattened",
    "BaseTransformer_flattened_only1pe",
    "BaseTransformer_flattened_ae",
    "BaseTransformer_flattened_aete",
    "BaseTransformer_flattened_AgentAwareAttention",
}

criterion_list = {"MSE": nn.MSELoss(), "L1": nn.L1Loss()}


# シード値
def seed_everything(seed=42):
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


# モデルを決定する関数
def modelDecision(model, look_back, dim, depth, heads, fc_dim, dim_head, dropout, emb_dropout):
    if "LSTM" in model:
        if model == "LSTM":
            from model.lstm.lstm import LSTMClassifier
        elif model == "LSTM_sensor_first":
            from model.lstm.lstm_input_sensor_first import LSTMClassifier
        elif model == "LSTM_3types":
            from model.lstm.lstm_input_3types import LSTMClassifier
        elif model == "LSTM_flattened":
            from model.lstm.lstm_input_flattened import LSTMClassifier
        elif model == "LSTM_individually":
            from model.lstm.lstm_input_individually import LSTMClassifier

        return LSTMClassifier(look_back=look_back, num_hidden_units=dim, num_layers=depth, dropout=dropout)

    if "BaseTransformer" in model:
        if model == "BaseTransformer":
            from model.transformer.base_transformer import BaseTransformer
        elif model == "BaseTransformer_only1pe":
            from model.transformer.base_transformer_only1pe import BaseTransformer
        elif model == "BaseTransformer_aete":
            from model.transformer.base_transformer_aete import BaseTransformer
        elif model == "BaseTransformer_sensor_first":
            from model.transformer.base_transformer_input_sensor_first import BaseTransformer
        elif model == "BaseTransformer_sensor_first_only1pe":
            from model.transformer.base_transformer_input_sensor_first_only1pe import BaseTransformer
        elif model == "BaseTransformer_sensor_first_no_pe":
            from model.transformer.base_transformer_input_sensor_first_no_pe import BaseTransformer
        elif model == "BaseTransformer_3types":
            from model.transformer.base_transformer_input_3types import BaseTransformer
        elif model == "BaseTransformer_3types_only1pe":
            from model.transformer.base_transformer_input_3types_only1pe import BaseTransformer
        elif model == "BaseTransformer_3types_ae":
            from model.transformer.base_transformer_input_3types_only1pe_ae import BaseTransformer
        elif model == "BaseTransformer_3types_aete":
            from model.transformer.base_transformer_input_3types_aete import BaseTransformer
        elif model == "BaseTransformer_3types_AgentAwareAttention":
            from model.transformer.base_transformer_input_3types_AgentAwareAttention import BaseTransformer
        elif model == "BaseTransformer_flattened":
            from model.transformer.base_transformer_input_flattened import BaseTransformer
        elif model == "BaseTransformer_flattened_only1pe":
            from model.transformer.base_transformer_input_flattened_only1pe import BaseTransformer
        elif model == "BaseTransformer_flattened_ae":
            from model.transformer.base_transformer_input_flattened_only1pe_ae import BaseTransformer
        elif model == "BaseTransformer_flattened_aete":
            from model.transformer.base_transformer_input_flattened_aete import BaseTransformer
        elif model == "BaseTransformer_flattened_AgentAwareAttention":
            from model.transformer.base_transformer_input_flattened_AgentAwareAttention import BaseTransformer

        return BaseTransformer(look_back=look_back, dim=dim, depth=depth, heads=heads, fc_dim=fc_dim, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)

    return None
