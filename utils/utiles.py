import torch
import torch.nn as nn
import random
import os
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error, mean_absolute_error


target_kW = {"ACDS_kW", "Comp_kW", "Eva_kW"}

predict_time_list = []

score_list_dict = {
    "ACDS_kW": {"ade": [], "fde": []},
    "Comp_kW": {"ade": [], "fde": []},
    "Eva_kW": {"ade": [], "fde": []},
}

test_score_list_dict = {
    "ACDS_kW": {"ade": [], "fde": []},
    "Comp_kW": {"ade": [], "fde": []},
    "Eva_kW": {"ade": [], "fde": []},
}

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
    "LSTM_input_sensor_first",
    "LSTM_input_3types",
    "LSTM_input_flattened",
    "LSTM_input_individually",
    "BaseTransformer",
    "BaseTransformer_only1pe",
    "BaseTransformer_aete",
    "BaseTransformer_input_sensor_first",
    "BaseTransformer_input_sensor_first_only1pe",
    "BaseTransformer_input_sensor_first_no_pe",
    "BaseTransformer_input_3types",
    "BaseTransformer_input_3types_only1pe",
    "BaseTransformer_input_3types_ae",
    "BaseTransformer_input_3types_aete",
    "BaseTransformer_input_3types_AgentAwareAttention",
    "BaseTransformer_input_flattened",
    "BaseTransformer_input_flattened_only1pe",
    "BaseTransformer_input_flattened_ae",
    "BaseTransformer_input_flattened_aete",
    "BaseTransformer_input_flattened_AgentAwareAttention",
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
        elif model == "BaseTransformer_aete":
            from model.transformer.base_transformer_aete import BaseTransformer
        elif model == "BaseTransformer_input_sensor_first":
            from model.transformer.base_transformer_input_sensor_first import BaseTransformer
        elif model == "BaseTransformer_input_sensor_first_only1pe":
            from model.transformer.base_transformer_input_sensor_first_only1pe import BaseTransformer
        elif model == "BaseTransformer_input_sensor_first_no_pe":
            from model.transformer.base_transformer_input_sensor_first_no_pe import BaseTransformer
        elif model == "BaseTransformer_input_3types":
            from model.transformer.base_transformer_input_3types import BaseTransformer
        elif model == "BaseTransformer_input_3types_only1pe":
            from model.transformer.base_transformer_input_3types_only1pe import BaseTransformer
        elif model == "BaseTransformer_input_3types_ae":
            from model.transformer.base_transformer_input_3types_only1pe_ae import BaseTransformer
        elif model == "BaseTransformer_input_3types_aete":
            from model.transformer.base_transformer_input_3types_aete import BaseTransformer
        elif model == "BaseTransformer_input_3types_AgentAwareAttention":
            from model.transformer.base_transformer_input_3types_AgentAwareAttention import BaseTransformer
        elif model == "BaseTransformer_input_flattened":
            from model.transformer.base_transformer_input_flattened import BaseTransformer
        elif model == "BaseTransformer_input_flattened_only1pe":
            from model.transformer.base_transformer_input_flattened_only1pe import BaseTransformer
        elif model == "BaseTransformer_input_flattened_ae":
            from model.transformer.base_transformer_input_flattened_only1pe_ae import BaseTransformer
        elif model == "BaseTransformer_input_flattened_aete":
            from model.transformer.base_transformer_input_flattened_aete import BaseTransformer
        elif model == "BaseTransformer_input_flattened_AgentAwareAttention":
            from model.transformer.base_transformer_input_flattened_AgentAwareAttention import BaseTransformer

        return BaseTransformer(look_back=look_back, dim=dim, depth=depth, heads=heads, fc_dim=fc_dim, dim_head=dim_head, dropout=dropout, emb_dropout=emb_dropout)

    return


def visualization(gt_array, pred_array, output_feature_name, output_feature_unit, case_name, case_path, is_normalized=False):
    for i in range(len(output_feature_name)):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.array(gt_array)[:, i], color="#e46409", label="gt")
        ax.plot(np.array(pred_array)[:, i], color="b", label="pred")
        ax.set_title(case_name)
        ax.set_xlabel("Time[s]")
        ax.set_ylabel(f"{output_feature_name[i]}[{output_feature_unit[i]}]")
        ax.legend(loc="best")
        plt.savefig(os.path.join(case_path, f"{output_feature_name[i]}.png"))
        plt.close()
    return


def evaluation(result_path, case_path, case_name, test_index, gt_array, pred_array, output_feature_name, output_feature_unit, num_fixed_data=8):
    for i in range(len(output_feature_name)):
        if output_feature_name[i] in target_kW:
            if test_index not in np.arange(0, num_fixed_data):
                ade = mean_absolute_error(np.array(gt_array)[:, i], np.array(pred_array)[:, i])
                fde = abs(gt_array[-1][i] - pred_array[-1][i])
                score_list_dict[output_feature_name[i]]["ade"].append(ade)
                score_list_dict[output_feature_name[i]]["fde"].append(fde)
            else:
                ade = mean_absolute_error(np.array(gt_array)[:, i], np.array(pred_array)[:, i])
                fde = abs(gt_array[-1][i] - pred_array[-1][i])
                test_score_list_dict[output_feature_name[i]]["ade"].append(ade)
                test_score_list_dict[output_feature_name[i]]["fde"].append(fde)
    visualization(gt_array, pred_array, output_feature_name, output_feature_unit, case_name, case_path)
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
