import numpy as np
import matplotlib.pyplot as plt
import os, mlflow

from sklearn.metrics import mean_absolute_error

target_kW = ["ACDS_kW", "Comp_kW", "Eva_kW", "Comp_OutP"]
target_kW_unit = ["kW", "kW", "kW", "MPa"]
evaluation_list = ["ade", "fde", "mde", "pde"]

score_list_dict = {
    "ACDS_kW": {"ade": [], "fde": [], "mde": [], "pde": []},
    "Comp_kW": {"ade": [], "fde": [], "mde": [], "pde": []},
    "Eva_kW": {"ade": [], "fde": [], "mde": [], "pde": []},
    "Comp_OutP": {"ade": [], "fde": [], "mde": [], "pde": []},
}

test_score_list_dict = {
    "ACDS_kW": {"ade": [], "fde": [], "mde": [], "pde": []},
    "Comp_kW": {"ade": [], "fde": [], "mde": [], "pde": []},
    "Eva_kW": {"ade": [], "fde": [], "mde": [], "pde": []},
    "Comp_OutP": {"ade": [], "fde": [], "mde": [], "pde": []},
}

target_kW_visualization = {
    "ACDS_kW": {"pred": [], "gt": []},
    "Comp_kW": {"pred": [], "gt": []},
    "Eva_kW": {"pred": [], "gt": []},
    "Comp_OutP": {"pred": [], "gt": []},
}

"""
    ade: average displacement error (Average of all time errors)
    fde: final displacement error (Error of the last time)
    mde: max displacement error (Maximum error of all times)
    pde: peek displacement error (Error of peak value )
"""

# Calculate Evaluation
def evaluation(test_index, gt_array, pred_array, output_feature_name, case_name, num_fixed_data=8):
    pred_array = pred_array.mean(axis=0)
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
                score_list_dict[output_feature_name[i]]["pde"].append(pde)
            else:
                test_score_list_dict[output_feature_name[i]]["ade"].append(ade)
                test_score_list_dict[output_feature_name[i]]["fde"].append(fde)
                test_score_list_dict[output_feature_name[i]]["mde"].append(mde)
                test_score_list_dict[output_feature_name[i]]["pde"].append(pde)

            case_evaluation_path = os.path.join("..", "result", "img", "original_scale", case_name)
            os.makedirs(case_evaluation_path, exist_ok=True)

            f = open(os.path.join(case_evaluation_path, "evaluation.txt"), "w")
            f.write(f"ade: {ade}\n")
            f.write(f"fde: {fde}\n")
            f.write(f"mde: {mde}\n")
            f.write(f"pde: {pde}\n")
            f.close()


# Save Evaluation
def save_evaluation(result_path):
    for target in target_kW:
        for evaluation in evaluation_list:
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


# Make Graph
def visualization(gt_array, pred_array, output_feature_name, output_feature_unit, case_name, result_path, is_normalized=False):
    img_path = os.path.join(result_path, "img", "normalized", case_name) if is_normalized else os.path.join(result_path, "img", "original_scale", case_name)
    os.makedirs(img_path, exist_ok=True)

    for i in range(len(output_feature_name)):
        mean = pred_array[:, :, i].mean(axis=0)
        std = pred_array[:, :, i].std(axis=0)
        fig = plt.figure()
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(np.array(gt_array)[:, i], color="#e46409", label="gt")
        ax.plot(mean, color="b", label="pred")
        ax.fill_between(np.arange(mean.shape[0]), mean + std, mean - std, alpha=0.2, color="b")
        ax.set_title(case_name)
        ax.set_xlabel("Time[s]")
        ax.set_ylabel(f"{output_feature_name[i]}[{output_feature_unit[i]}]")
        ax.legend(loc="best")
        plt.savefig(os.path.join(img_path, f"{output_feature_name[i]}.png"))
        plt.close()
        if output_feature_name[i] in target_kW:
            target_kW_visualization[output_feature_name[i]]["pred"] = np.array(pred_array)[:, :, i]
            target_kW_visualization[output_feature_name[i]]["gt"] = np.array(gt_array)[:, i]

    # For PowerPoint Slide
    grf_row = 4
    grf_col = 1
    fig = plt.figure(figsize=(grf_col * 8, grf_row * 4))
    ax_list = []
    for j in range(grf_row):
        for i in range(grf_col):
            if (j * grf_col + i) < len(target_kW):
                mean = target_kW_visualization[target_kW[j * grf_col + i]]["pred"].mean(axis=0)
                std = target_kW_visualization[target_kW[j * grf_col + i]]["pred"].std(axis=0)
                ax_list.append(fig.add_subplot(grf_row, grf_col, j * grf_col + i + 1))
                ax_list[j * grf_col + i].plot(target_kW_visualization[target_kW[j * grf_col + i]]["gt"], color="#e46409", label="gt")
                ax_list[j * grf_col + i].plot(mean, color="b", label="pred")
                ax_list[j * grf_col + i].fill_between(np.arange(mean.shape[0]), mean + std, mean - std, alpha=0.2, color="b")
                ax_list[j * grf_col + i].set_xlabel(f"Time[s]")
                ax_list[j * grf_col + i].set_ylabel(f"{target_kW[j * grf_col + i]}[{target_kW_unit[j * grf_col + i]}]")
                ax_list[j * grf_col + i].legend(loc="best")

    fig.tight_layout()
    plt.savefig(os.path.join(img_path, f"target_kW.png"))
    plt.close()


# Make AttentionMap Graph
def attention_visualization(args, attn_all, result_path, case_name):
    attn_all = attn_all.mean(axis=0)
    for i in range(len(attn_all)):
        if i % 50 == 0:
            attn = attn_all[i]
            for depth in range(args.depth):
                img_path = os.path.join(result_path, "attention", case_name, str(i + 1))
                os.makedirs(img_path, exist_ok=True)
                """
                for head in range(args.heads):
                    head_path = os.path.join(img_path, str(depth+1))
                    os.makedirs(head_path, exist_ok=True)
                    fig = plt.figure()
                    plt.imshow(np.array(attn[depth, head, :, :]), cmap="Reds")
                    plt.colorbar()
                    plt.savefig(os.path.join(head_path, f"attention_heads{head+1}.png"))
                    plt.close()
                """
                attn_mean = np.mean(attn[depth], axis=0)
                fig = plt.figure()
                plt.imshow(np.array(attn_mean), cmap="Reds")
                plt.colorbar()
                plt.savefig(os.path.join(img_path, f"attention_mean_depth{depth+1}.png"))
                plt.close()
