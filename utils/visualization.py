import numpy as np
import matplotlib.pyplot as plt
import os, mlflow

from sklearn.metrics import mean_absolute_error

class Eva():
    def __init__(self, data):
        self.pred_name = data["pred_name"]
        self.pred_unit = data["pred_unit"]
        self.target_name = data["target_name"]
        self.target_unit = data["target_unit"]
        
        self.evaluation_list = ["ade", "fde", "mde", "pde"]
        self.eva_all = []

        """
            ade: average displacement error (Average of all time errors)
            fde: final displacement error (Error of the last time)
            mde: max displacement error (Maximum error of all times)
            pde: peak displacement error (Error of peak value)
        """


    # Calculate Evaluation
    def evaluation(self, in_len, gt_array, pred_array, case_name):
        eva_case = []
        for i in range(len(self.pred_name)):
            ade = mean_absolute_error(np.array(gt_array)[in_len:, i], np.array(pred_array)[in_len:, i])
            fde = abs(gt_array[-1][i] - pred_array[-1][i])
            mde = max(abs(np.array(gt_array)[in_len:, i] - np.array(pred_array)[in_len:, i]))
            pde = abs(max(abs(np.array(gt_array)[in_len:, i])) - max(abs(np.array(pred_array)[in_len:, i])))

            eva_case.append(ade)
            eva_case.append(fde)
            eva_case.append(mde)
            eva_case.append(pde)

            case_evaluation_path = os.path.join("..", "result", "vis", "original_scale", case_name, "evaluation")
            os.makedirs(case_evaluation_path, exist_ok=True)

            f = open(os.path.join(case_evaluation_path, f"{self.pred_name[i]}.txt"), "w")
            f.write(f"ade: {ade}\n")
            f.write(f"fde: {fde}\n")
            f.write(f"mde: {mde}\n")
            f.write(f"pde: {pde}\n")
            f.close()
                
        self.eva_all.append(eva_case)


    # Save Evaluation
    def save_evaluation(self, result_path):
        for i, target in enumerate(self.pred_name):
            for j, evaluation in enumerate(self.evaluation_list):
                np_array = np.array(self.eva_all)[:, j + len(self.evaluation_list) * i]
                mlflow.log_metric(f"{target}_{evaluation.upper()}_mean", np.mean(np_array))

                evaluation_path = os.path.join(result_path, "evaluation")
                os.makedirs(evaluation_path, exist_ok=True)

                f = open(os.path.join(evaluation_path, f"{target}_{evaluation.upper()}.txt"), "w")
                f.write(f"max: {np.max(np_array)}\n")
                f.write(f"min: {np.min(np_array)}\n")
                f.write(f"mean: {np.mean(np_array)}\n")
                f.write(f"median: {np.median(np_array)}\n")
                f.close()


    # Make Graph
    def visualization(self, gt_array, pred_array, case_name, result_path, is_normalized=False):
        img_path = os.path.join(result_path, "vis", "normalized", case_name, "img") if is_normalized else os.path.join(result_path, "vis", "original_scale", case_name, "img")
        os.makedirs(img_path, exist_ok=True)

        for i in range(len(self.pred_name)):
            fig = plt.figure()
            ax = fig.add_subplot(1, 1, 1)
            ax.plot(np.array(gt_array)[:, i], color="#e46409", label="gt")
            ax.plot(np.array(pred_array)[:, i], color="b", label="pred")
            ax.set_title(case_name, fontsize=16)
            ax.set_xlabel("Time[s]", fontsize=16)
            ax.set_ylabel(f"{self.pred_name[i]}[{self.pred_unit[i]}]", fontsize=16)
            ax.set_yticks([])
            ax.set_xticks([0, 600, 1200])
            ax.legend(loc="best", fontsize=16)
            plt.savefig(os.path.join(img_path, f"{self.pred_name[i]}.png"))
            plt.close()

        # For PowerPoint Slide
        grf_row = 4
        grf_col = 1
        fig = plt.figure(figsize=(grf_col * 8, grf_row * 4))
        ax_list = []
        for j in range(grf_row):
            for i in range(grf_col):
                if (j * grf_col + i) < len(self.target_name):
                    ax_list.append(fig.add_subplot(grf_row, grf_col, j * grf_col + i + 1))
                    ax_list[j * grf_col + i].plot(np.array(gt_array)[:, self.pred_name.index(self.target_name[j*grf_col+i])], color="#e46409", label="gt")
                    ax_list[j * grf_col + i].plot(np.array(pred_array)[:, self.pred_name.index(self.target_name[j*grf_col+i])], color="b", label="pred")
                    ax_list[j * grf_col + i].set_xlabel(f"Time[s]"), 
                    ax_list[j * grf_col + i].set_ylabel(f"{self.target_name[j * grf_col + i]}[{self.target_unit[j * grf_col + i]}]")
                    ax_list[j * grf_col + i].legend(loc="best")

        fig.tight_layout()
        plt.savefig(os.path.join(img_path, f"target_kW.png"))
        plt.close()


    # Make AttentionMap Graph
    def attention_visualization(self, args, attn_all, result_path, case_name):
        for i in range(len(attn_all)):
            if i % 50 == 0:
                attn = attn_all[i]
                for depth in range(args.e_layers):
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


def print_model_summary(args, device, len_train_index, len_val_index):
    print("\n\nTrain Start")
    print("----------------------------------------------")
    print(f"Device      : {str.upper(device)}")
    print(f"Train Case  : {len_train_index}")
    print(f"Val Case    : {len_val_index}")
    print(f"Criterion   : {args.criterion}")
    print(f"Batch Size  : {args.bs}")
    print(f"in_len      : {args.in_len}")
    print(f"Model       : {args.model}")
    print(f" - e_layers  : {args.e_layers}")
    print(f" - d_model   : {args.d_model}")
    print(f" - dropout   : {args.dropout}")
    if ("BaseTransformer" in args.model) or ("Crossformer" in args.model) or ("DeepO" in args.model):
        print(f" - num heads : {args.n_heads}")
        print(f" - dim of ff : {args.d_ff}")
        if "Crossformer" in args.model:
            print(f" - seg len   : {args.seg_len}")
            print(f" - win size  : {args.win_size}")
            print(f" - factor num: {args.factor}")
        if "DeepO" in args.model:
            print(f" - trunk layers : {args.trunk_layers}")
            print(f" - trunk dim    : {args.trunk_d_model}")