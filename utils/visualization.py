import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import os, mlflow

from einops import rearrange
from sklearn.metrics import mean_absolute_error

target_kW = {"ACDS_kW", "Comp_kW", "Eva_kW", "Comp_OutP"}
target_kW_unit = {"kW", "kW", "kW", "MPa"}

score_list_dict = {
    "ACDS_kW": {"ade": [], "fde": [], "mde": []},
    "Comp_kW": {"ade": [], "fde": [], "mde": []},
    "Eva_kW": {"ade": [], "fde": [], "mde": []},
    "Comp_OutP": {"ade": [], "fde": [], "mde": []},
}

test_score_list_dict = {
    "ACDS_kW": {"ade": [], "fde": [], "mde": []},
    "Comp_kW": {"ade": [], "fde": [], "mde": []},
    "Eva_kW": {"ade": [], "fde": [], "mde": []},
    "Comp_OutP": {"ade": [], "fde": [], "mde": []},
}

target_kW_visualization = {
    "ACDS_kW": {"pred": [], "gt": []},
    "Comp_kW": {"pred": [], "gt": []},
    "Eva_kW": {"pred": [], "gt": []},
    "Comp_OutP": {"pred": [], "gt": []},
}

"""
    ade: average displacement error (全時刻の誤差の平均)
    fde: final displacement error (最終時刻の誤差)
    mde: max displacement error (全時刻の誤差の最大値)
    pde: peek displacement error (ピーク値の誤差)
"""

# 評価を計算する関数
def evaluation(test_index, gt_array, pred_array, output_feature_name, num_fixed_data=8):
    for i in range(len(output_feature_name)):
        if output_feature_name[i] in target_kW:
            ade = mean_absolute_error(np.array(gt_array)[:, i], np.array(pred_array)[:, i])
            fde = abs(gt_array[-1][i] - pred_array[-1][i])
            mde = max(abs(np.array(gt_array)[:, i] - np.array(pred_array)[:, i]))
            pde = abs(max(abs(np.array(gt_array)[:, i])) - max(abs(np.array(pred_array)[:, i])))
            if test_index not in np.arange(0, num_fixed_data):
                score_list_dict[output_feature_name[i]]["ade"].append(ade)
                score_list_dict[output_feature_name[i]]["fde"].append(fde)
                score_list_dict[output_feature_name[i]]["mde"].append(mde)
            else:
                test_score_list_dict[output_feature_name[i]]["ade"].append(ade)
                test_score_list_dict[output_feature_name[i]]["fde"].append(fde)
                test_score_list_dict[output_feature_name[i]]["mde"].append(mde)


# 計算した評価を保存する関数
def save_evaluation(result_path):
    for target in target_kW:
        for evaluation in ["ade", "fde", "mde"]:
            np_array = np.array(score_list_dict[target][evaluation])
            np_array_test = np.array(test_score_list_dict[target][evaluation])
            mlflow.log_metric(f"{target}_{evaluation.upper()}_mean", np.mean(np_array))
            mlflow.log_metric(f"test_{target}_{evaluation.upper()}_mean", np.mean(np_array_test))

            evaluation_path = os.path.join(result_path, "evaluation")
            os.makedirs(evaluation_path, exist_ok=True)

            f = open(os.path.join(evaluation_path, f"{target}_{evaluation.upper()}.txt"), "w")
            f.write(f"max: {np.max(np_array)}\n")
            f.write(f"min: {np.min(np_array)}\n")
            f.write(f"mean: {np.mean(np_array)}\n")
            f.write(f"median: {np.median(np_array)}\n")
            f.close()

            f = open(os.path.join(evaluation_path, f"test_{target}_{evaluation.upper()}.txt"), "w")
            f.write(f"max: {np.max(np_array_test)}\n")
            f.write(f"min: {np.min(np_array_test)}\n")
            f.write(f"mean: {np.mean(np_array_test)}\n")
            f.write(f"median: {np.median(np_array_test)}\n")
            f.close()


# グラフを作る関数
def visualization(gt_array, pred_array, output_feature_name, output_feature_unit, case_name, result_path, is_normalized=False):
    img_path = os.path.join(result_path, "img", "normalized", case_name) if is_normalized else os.path.join(result_path, "img", "original_scale", case_name)
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
        if output_feature_name[i] in target_kW:
            target_kW_visualization[output_feature_name[i]]["pred"] = np.array(pred_array)[:, i]
            target_kW_visualization[output_feature_name[i]]["gt"] = np.array(gt_array)[:, i]

    fig = plt.figure(figsize=(8, 16))
    ax1 = fig.add_subplot(4, 1, 1)
    ax2 = fig.add_subplot(4, 1, 2)
    ax3 = fig.add_subplot(4, 1, 3)
    ax4 = fig.add_subplot(4, 1, 4)

    ax1.plot(np.array(target_kW_visualization["ACDS_kW"]["gt"]), color="#e46409", label="gt")
    ax1.plot(np.array(target_kW_visualization["ACDS_kW"]["pred"]), color="b", label="pred")
    ax2.plot(np.array(target_kW_visualization["Comp_kW"]["gt"]), color="#e46409", label="gt")
    ax2.plot(np.array(target_kW_visualization["Comp_kW"]["pred"]), color="b", label="pred")
    ax3.plot(np.array(target_kW_visualization["Eva_kW"]["gt"]), color="#e46409", label="gt")
    ax3.plot(np.array(target_kW_visualization["Eva_kW"]["pred"]), color="b", label="pred")
    ax4.plot(np.array(target_kW_visualization["Comp_OutP"]["gt"]), color="#e46409", label="gt")
    ax4.plot(np.array(target_kW_visualization["Comp_OutP"]["pred"]), color="b", label="pred")

    ax1.set_xlabel("Time[s]")
    ax1.set_ylabel("ACDS_kW[kW]")
    ax2.set_xlabel("Time[s]")
    ax2.set_ylabel("Comp_kW[kW]")
    ax3.set_xlabel("Time[s]")
    ax3.set_ylabel("Eva_kW[kW]")
    ax4.set_xlabel("Time[s]")
    ax4.set_ylabel("Comp_OutP[MPa]")

    ax1.legend(loc="best")
    ax2.legend(loc="best")
    ax3.legend(loc="best")
    ax4.legend(loc="best")

    fig.tight_layout()
    plt.savefig(os.path.join(img_path, f"target_kW.png"))
    plt.close()


# アテンションマップを可視化する関数
def attention_visualization(attn_all, depth, heads, result_path, case_name):
    for i in range(len(attn_all)):
        if i % 50 == 0:
            attn = attn_all[i]
            for j in range(depth):
                img_path = os.path.join(result_path, "attention", case_name, str(i + 1))
                os.makedirs(img_path, exist_ok=True)
                """
                for head in range(heads):
                    fig = plt.figure()
                    plt.imshow(np.array(attn[j, head, :, :]), cmap="Reds")
                    plt.colorbar()
                    plt.savefig(os.path.join(img_path, f"attention_depth{j+1}_heads{head+1}.png"))
                    plt.close()
                """
                attn_mean = np.mean(attn[j], axis=0)
                fig = plt.figure()
                plt.imshow(np.array(attn_mean), cmap="Reds")
                plt.colorbar()
                plt.savefig(os.path.join(img_path, f"attention_mean_depth{j+1}.png"))
                plt.close()
