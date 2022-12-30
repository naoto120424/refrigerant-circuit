import torch
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import shutil
import argparse
import time
import datetime
from utils.dataloader import *
from utils.utiles import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error


def main():
    parser = argparse.ArgumentParser(description="Mazda Refrigerant Circuit Project")
    parser.add_argument("--e_name", type=str, default="Mazda Refrigerant Circuit Tutorial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bs", type=int, default=32)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--look_back", type=int, default=20)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--fc_dim", type=int, default=2048)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--emb_dropout", type=float, default=0.1)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--model", type=str, default="BaseTransformer")
    parser.add_argument("--criterion", type=str, default="MSE")
    args = parser.parse_args()

    if args.debug:
        epoch_num = 3
    else:
        epoch_num = args.epoch

    mlflow.set_tracking_uri("../mlflow_experiment")
    mlflow.set_experiment(args.e_name)
    mlflow.start_run()
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("epoch num", epoch_num)
    mlflow.log_param("batch size", args.bs)
    mlflow.log_param("look back", args.look_back)
    mlflow.log_param("debug", args.debug)
    mlflow.log_param("model", args.model)
    mlflow.log_param("criterion", args.criterion)
    mlflow.log_param("dim", args.dim)
    mlflow.log_param("depth", args.depth)
    mlflow.log_param("dropout", args.dropout)
    if "BaseTransformer" in args.model:
        mlflow.log_param("heads", args.heads)
        mlflow.log_param("fc_dim", args.fc_dim)
        mlflow.log_param("dim_head", args.dim_head)
        mlflow.log_param("emb_dropout", args.emb_dropout)

    seed_everything(seed=args.seed)

    device = deviceChecker()
    data = load_data(look_back=args.look_back)

    num_fixed_data = 8
    num_control_features = 6

    train_index_list, test_index_list = train_test_split(np.arange(num_fixed_data, len(data["inp"])), test_size=200)
    train_index_list, val_index_list = train_test_split(train_index_list, test_size=100)
    test_index_list = np.append(test_index_list, np.arange(0, num_fixed_data))

    train_dataset, mean_list, std_list = create_dataset(data, train_index_list, is_train=True)
    val_dataset, _, _ = create_dataset(data, val_index_list, is_train=False, mean_list=mean_list, std_list=std_list)

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

    if args.model in model_list:
        model = modelDecision(args.model, args.look_back, args.dim, args.depth, args.heads, args.fc_dim, args.dim_head, args.dropout, args.emb_dropout)
    else:
        print("unknown model name")
        return

    if args.criterion in criterion_list:
        criterion = criterion_list[args.criterion]
    else:
        print("unknown criterion name")
        return

    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    best_loss = 1000.0
    print("\n\nTrain Start")
    print("----------------------------------------------")
    print(f"Device      : {str.upper(device)}")
    print(f"Model       : {args.model}")
    print(f" -dim        : {args.dim}")
    print(f" -layer      : {args.depth}")
    print(f" -dropout    : {args.dropout}")
    if "BaseTransformer" in args.model:
        print(f" -heads      : {args.heads}")
        print(f" -fc_dim     : {args.fc_dim}")
        print(f" -dim_head   : {args.dim_head}")
        print(f" -emb_dropout: {args.emb_dropout}")
    print(f"Criterion   : {args.criterion}")
    print(f"Look Back   : {args.look_back}")
    print(f"Batch Size  : {args.bs}")
    print(f"train case  : {len(train_index_list)}")
    print(f"val case    : {len(val_index_list)}")
    train_start_time = time.perf_counter()
    for epoch in range(1, epoch_num + 1):
        print("----------------------------------------------")
        print(f"[Epoch {epoch}/{epoch_num}]")
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            inp = batch["inp"].to(device)
            spec = batch["spec"].to(device)
            gt = batch["gt"].to(device)

            pred = model(inp, spec)
            loss = criterion(pred, gt)
            loss.backward()
            epoch_loss += loss.item() * inp.size(0)
            optimizer.step()
        epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Train Loss: {epoch_loss}")
        mlflow.log_metric(f"train loss", epoch_loss, step=epoch)

        with torch.no_grad():
            model.eval()
            epoch_test_error = 0
            for batch in tqdm(val_dataloader):
                inp = batch["inp"].to(device)
                spec = batch["spec"].to(device)
                gt = batch["gt"].to(device)

                pred = model(inp, spec)
                test_error = torch.mean(torch.abs(gt - pred))
                epoch_test_error += test_error.item() * inp.size(0)

            epoch_test_error = epoch_test_error / len(val_dataloader)
            print(f"Val Loss: {epoch_test_error}")
            mlflow.log_metric(f"val loss", epoch_test_error, step=epoch)

        result_path = os.path.join("..", "result")
        os.makedirs(result_path, exist_ok=True)

        if epoch_test_error < best_loss:
            best_epoch_num = epoch
            best_loss = epoch_test_error
            print('This is the best model. Save to "best_model.pth".')
            model_path = os.path.join(result_path, "best_model.pth")
            torch.save(model.state_dict(), model_path)
    mlflow.log_metric(f"best epoch num", best_epoch_num)
    train_end_time = time.perf_counter()
    train_time = datetime.timedelta(seconds=(train_end_time - train_start_time))
    mlflow.log_metric(f"train time", train_end_time - train_start_time)

    print("----------------------------------------------")
    print(f"Train Time: {train_time}")

    with torch.no_grad():
        print(f"\n\nTest Start. Case Num: {len(test_index_list)}")
        print("----------------------------------------------")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        for test_index in tqdm(test_index_list):
            if test_index in np.arange(0, num_fixed_data):
                case_name = list(fixed_case_list)[test_index]
            else:
                case_name = f"case{str(test_index-num_fixed_data+1).zfill(4)}"
            case_path = os.path.join(result_path, "img", case_name)
            os.makedirs(case_path, exist_ok=True)
            inp_data = data["inp"][test_index]
            spec_data = data["spec"][test_index]
            gt_data = data["gt"][test_index]

            scaling_input_data = inp_data[0].copy()
            scaling_spec_data = spec_data.copy()
            for i in range(scaling_input_data.shape[1]):
                scaling_input_data[:, i] = (scaling_input_data[:, i] - mean_list[i]) / std_list[i]
            for i in range(scaling_spec_data.shape[1]):
                scaling_spec_data[:, i] = (scaling_spec_data[:, i] - mean_list[i]) / std_list[i]

            start_time = time.perf_counter()
            for i in range(scaling_spec_data.shape[0]):
                input = torch.from_numpy(scaling_input_data[i : i + args.look_back].astype(np.float32)).clone().unsqueeze(0).to(device)
                spec = torch.from_numpy(scaling_spec_data[i].astype(np.float32)).clone().unsqueeze(0).to(device)
                scaling_pred_data = model(input, spec).detach().to("cpu").numpy().copy()[0]
                new_scaling_input_data = np.append(scaling_spec_data[i], scaling_pred_data)[np.newaxis, :]
                scaling_input_data = np.concatenate([scaling_input_data, new_scaling_input_data], axis=0)
            end_time = time.perf_counter()
            predict_time_list.append(end_time - start_time)

            pred_data = scaling_input_data.copy()
            for i in range(scaling_input_data.shape[1]):
                pred_data[:, i] = pred_data[:, i] * std_list[i] + mean_list[i]
            pred_output_data = pred_data[:, num_control_features:]

            output_feature_name = data["feature_name"][num_control_features + 1 :]
            output_feature_unit = data["feature_unit"][num_control_features + 1 :]
            gt_output_data = []
            for inp in inp_data[0]:
                gt_output_data.append(inp[num_control_features:])
            for gt in gt_data:
                gt_output_data.append(gt)

            for i in range(len(output_feature_name)):
                if output_feature_name[i] in target_kW:
                    if test_index not in np.arange(0, num_fixed_data):
                        ade = mean_absolute_error(np.array(gt_output_data)[:, i], np.array(pred_output_data)[:, i])
                        fde = abs(gt_output_data[-1][i] - pred_output_data[-1][i])
                        score_list_dict[output_feature_name[i]]["ade"].append(ade)
                        score_list_dict[output_feature_name[i]]["fde"].append(fde)
                    else:
                        ade = mean_absolute_error(np.array(gt_output_data)[:, i], np.array(pred_output_data)[:, i])
                        fde = abs(gt_output_data[-1][i] - pred_output_data[-1][i])
                        test_score_list_dict[output_feature_name[i]]["ade"].append(ade)
                        test_score_list_dict[output_feature_name[i]]["fde"].append(fde)
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(np.array(gt_output_data)[:, i], color="#e46409", label="gt")
                ax.plot(np.array(pred_output_data)[:, i], color="b", label="pred")
                ax.set_title(case_name)
                ax.set_xlabel("Time[s]")
                ax.set_ylabel(f"{output_feature_name[i]}[{output_feature_unit[i]}]")
                ax.legend(loc="best")
                plt.savefig(os.path.join(case_path, f"{output_feature_name[i]}.png"))
                plt.close()

        for target in target_kW:
            for evaluation in ["ade", "fde"]:
                np_array = np.array(score_list_dict[target][evaluation])
                np_array_test = np.array(test_score_list_dict[target][evaluation])
                mlflow.log_metrics(
                    {
                        f"{target}_{evaluation.upper()}_max": np.max(np_array),
                        f"{target}_{evaluation.upper()}_min": np.min(np_array),
                        f"{target}_{evaluation.upper()}_mean": np.mean(np_array),
                        f"{target}_{evaluation.upper()}_median": np.median(np_array),
                    }
                )
                mlflow.log_metrics(
                    {
                        f"test_{target}_{evaluation.upper()}_max": np.max(np_array_test),
                        f"test_{target}_{evaluation.upper()}_min": np.min(np_array_test),
                        f"test_{target}_{evaluation.upper()}_mean": np.mean(np_array_test),
                        f"test_{target}_{evaluation.upper()}_median": np.median(np_array_test),
                    }
                )
                """
                evaluation_path = os.path.join(result_path, "evaluation")
                os.makedirs(evaluation_path, exist_ok=True)
                f = open(os.path.join(evaluation_path, f"{target}_{evaluation.upper()}.txt"), "w")
                f.write(f"max: {np.max(np_array)}")
                f.write(f"min: {np.min(np_array)}")
                f.write(f"mean: {np.mean(np_array)}")
                f.write(f"median: {np.median(np_array)}")
                f.close()
                f = open(os.path.join(evaluation_path, f"test_{target}_{evaluation.upper()}.txt"), "w")
                f.write(f"max: {np.max(np_array_test)}")
                f.write(f"min: {np.min(np_array_test)}")
                f.write(f"mean: {np.mean(np_array_test)}")
                f.write(f"median: {np.median(np_array_test)}")
                f.close()
                """

    mlflow.log_metric(f"predict time mean", np.mean(predict_time_list))
    mlflow.log_artifacts(local_dir=result_path, artifact_path="result")
    shutil.rmtree(result_path)
    mlflow.end_run()
    print(f"Predict Time: {np.mean(predict_time_list)} [s]")
    print("----------------------------------------------\n")
    print("Experiment End")


if __name__ == "__main__":
    main()
