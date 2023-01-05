import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import random, os, mlflow

from einops import rearrange
from sklearn.metrics import mean_absolute_error


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

    return None


# 評価を計算する関数
def evaluation(test_index, gt_array, pred_array, output_feature_name, num_fixed_data=8, debug=False):
    for i in range(len(output_feature_name)):
        if output_feature_name[i] in target_kW:
            ade = mean_absolute_error(np.array(gt_array)[:, i], np.array(pred_array)[:, i])
            fde = abs(gt_array[-1][i] - pred_array[-1][i])
            if test_index not in np.arange(0, num_fixed_data) or debug:
                score_list_dict[output_feature_name[i]]["ade"].append(ade)
                score_list_dict[output_feature_name[i]]["fde"].append(fde)
            else:
                test_score_list_dict[output_feature_name[i]]["ade"].append(ade)
                test_score_list_dict[output_feature_name[i]]["fde"].append(fde)


# 計算した評価を保存する関数
def save_evaluation(result_path, debug=False):
    for target in target_kW:
        for evaluation in ["ade", "fde"]:
            np_array = np.array(score_list_dict[target][evaluation])
            mlflow.log_metric(f"{target}_{evaluation.upper()}_mean", np.mean(np_array))

            evaluation_path = os.path.join(result_path, "evaluation")
            os.makedirs(evaluation_path, exist_ok=True)

            f = open(os.path.join(evaluation_path, f"{target}_{evaluation.upper()}.txt"), "w")
            f.write(f"max: {np.max(np_array)}\n")
            f.write(f"min: {np.min(np_array)}\n")
            f.write(f"mean: {np.mean(np_array)}\n")
            f.write(f"median: {np.median(np_array)}\n")
            f.close()

            if not debug:
                np_array_test = np.array(test_score_list_dict[target][evaluation])
                mlflow.log_metric(f"test_{target}_{evaluation.upper()}_mean", np.mean(np_array_test))
                f = open(os.path.join(evaluation_path, f"test_{target}_{evaluation.upper()}.txt"), "w")
                f.write(f"max: {np.max(np_array_test)}\n")
                f.write(f"min: {np.min(np_array_test)}\n")
                f.write(f"mean: {np.mean(np_array_test)}\n")
                f.write(f"median: {np.median(np_array_test)}\n")
                f.close()


# グラフを作る関数
def visualization(gt_array, pred_array, output_feature_name, output_feature_unit, case_name, img_path, is_normalized=False):
    img_path = os.path.join(img_path, "normalized", case_name) if is_normalized else os.path.join(img_path, "original_scale", case_name)
    os.makedirs(img_path, exist_ok=True)

    for i in range(len(output_feature_name)):
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.array(gt_array)[:, i], color="#e46409", label="gt")
        ax.plot(np.array(pred_array)[:, i], color="b", label="pred")
        ax.set_title(case_name)
        ax.set_xlabel("Time[s]")
        ax.set_ylabel(f"{output_feature_name[i]}[{output_feature_unit[i]}]")
        ax.legend(loc="best")
        plt.savefig(os.path.join(img_path, f"{output_feature_name[i]}.png"))
        plt.close()


# アテンションマップを可視化する関数
def attention_visualization(model, result_path, model_name, dim, dim_heads, heads, look_back, num_control_features, num_all_features, device):
    scale = dim_heads**-0.5
    if "input_3types" in model_name:
        num_agent = 3
    elif "input_flattened" in model_name:
        num_agent = num_all_features
    else:
        num_agent = 1
    x = torch.rand(1, look_back * num_agent + num_control_features, dim).to(device)

    i = 0
    for key, value in model.state_dict().items():
        if "to_qkv" in key:
            i = i + 1
            value_weight = nn.Parameter(value).to(device)
            qkv = nn.functional.linear(x, value_weight).chunk(3, dim=-1)
            q, k, v = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=heads), qkv)
            dots = torch.matmul(q, k.transpose(-1, -2)) * scale
            attend = nn.Softmax(dim=-1)

            if "AgentAwareAttention" in model_name:
                key_to_qk_self = key.replace("to_qkv", "to_qk_self")
                value_to_qk_self = model.state_dict()[key_to_qk_self]
                value_self_weight = nn.Parameter(value_to_qk_self).to(device)
                qk_self = nn.functional.linear(x, value_self_weight).chunk(2, dim=-1)
                q_self, k_self = map(lambda t: rearrange(t, "b n (h d) -> b h n d", h=heads), qk_self)
                dots_self = torch.matmul(q_self, k_self.transpose(-1, -2)) * scale
                
                attn_mask = torch.eye(num_agent)
                attn_mask = attn_mask.repeat_interleave(look_back, dim=1)
                attn_mask = attn_mask.repeat_interleave(look_back, dim=0)
                attn_mask = torch.cat([attn_mask, torch.zeros(attn_mask.size(0), num_control_features)], dim=1)
                attn_mask = torch.cat([attn_mask, torch.zeros(num_control_features, attn_mask.size(1))], dim=0)
                attn_mask = attn_mask.unsqueeze(0).unsqueeze(0).to(device)
                
                dots = attn_mask * dots_self + (1 - attn_mask) * dots
                
            attn = attend(dots)
            attn_img = attn.detach().to("cpu").numpy().copy()[0]
            img_path = os.path.join(result_path, "attention", str(i))
            os.makedirs(img_path, exist_ok=True)
            for head in range(heads):
                fig = plt.figure()
                plt.imshow(np.array(attn_img[head, :, :]), cmap="Reds")
                plt.colorbar()
                plt.savefig(os.path.join(img_path, f"attention_heads{head+1}.png"))
                plt.close()

            