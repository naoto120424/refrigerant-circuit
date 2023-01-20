import torch
from torch.utils.data import DataLoader
import numpy as np
import time, datetime
import mlflow, shutil, argparse

from MCdropout.utils import *
from MCdropout.visualization import *
from utils.dataloader import *
from utils.earlystopping import EarlyStopping
from sklearn.model_selection import train_test_split


def main():
    parser = argparse.ArgumentParser(description="Mazda Refrigerant Circuit Project")
    parser.add_argument("--e_name", type=str, default="Mazda Refrigerant Circuit Tutorial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bs", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=10000)
    parser.add_argument("--look_back", type=int, default=5)
    parser.add_argument("--dim", type=int, default=512)
    parser.add_argument("--depth", type=int, default=3)
    parser.add_argument("--heads", type=int, default=8)
    parser.add_argument("--dim_head", type=int, default=64)
    parser.add_argument("--fc_dim", type=int, default=2048)
    parser.add_argument("--dropout", type=float, default=0.1)
    parser.add_argument("--emb_dropout", type=float, default=0.1)
    parser.add_argument("--debug", type=bool, default=False)
    parser.add_argument("--model", type=str, default="BaseTransformer")
    parser.add_argument("--criterion", type=str, default="MSE")
    parser.add_argument("--patience", type=int, default=7)
    parser.add_argument("--delta", type=float, default=1e-3)
    args = parser.parse_args()

    mlflow.set_tracking_uri("../../mlflow_experiment")
    mlflow.set_experiment(args.e_name)
    mlflow.start_run()
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("batch size", args.bs)
    mlflow.log_param("look back", args.look_back)
    mlflow.log_param("epoch", args.epoch)
    mlflow.log_param("debug", args.debug)
    mlflow.log_param("model", args.model)
    mlflow.log_param("criterion", args.criterion)
    mlflow.log_param("patience", args.patience)
    mlflow.log_param("delta", args.delta)
    mlflow.log_param("dim", args.dim)
    mlflow.log_param("depth", args.depth)
    mlflow.log_param("dropout", args.dropout)
    if "BaseTransformer" in args.model:
        mlflow.log_param("heads", args.heads)
        mlflow.log_param("dim_head", args.dim_head)
        mlflow.log_param("fc_dim", args.fc_dim)
        mlflow.log_param("emb_dropout", args.emb_dropout)

    seed_everything(seed=args.seed)
    data = load_data(look_back=args.look_back)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.debug:
        train_index_list, test_index_list = train_test_split(np.arange(CFG.NUM_FIXED_DATA, len(data["inp"])), test_size=200)
        train_index_list, val_index_list = train_test_split(train_index_list, test_size=100)
    else:
        train_index_list, test_index_list = train_test_split(np.arange(CFG.NUM_FIXED_DATA, CFG.NUM_FIXED_DATA + 100), test_size=10)
        train_index_list, val_index_list = train_test_split(train_index_list, test_size=10)

    test_index_list = np.append(test_index_list, np.arange(CFG.NUM_FIXED_DATA))

    train_dataset, mean_list, std_list = create_dataset(data, train_index_list, is_train=True)
    val_dataset, _, _ = create_dataset(data, val_index_list, is_train=False, mean_list=mean_list, std_list=std_list)

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True)

    model = modelDecision(args) if args.model in model_list else print(f"unknown model name")
    criterion = criterion_list[args.criterion] if args.criterion in criterion_list else print(f"unknown criterion name")

    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, verbose=True)
    epoch_num = args.epoch if not args.debug else 3

    print("\n\nTrain Start")
    print("----------------------------------------------")
    print(f"Device      : {str.upper(device)}")
    print(f"Model       : {args.model}")
    print(f" -dim        : {args.dim}")
    print(f" -layer      : {args.depth}")
    print(f" -dropout    : {args.dropout}")
    if "BaseTransformer" in args.model:
        print(f" -heads      : {args.heads}")
        print(f" -dim_head   : {args.dim_head}")
        print(f" -fc_dim     : {args.fc_dim}")
        print(f" -emb_dropout: {args.emb_dropout}")
    print(f"Criterion   : {args.criterion}")
    print(f"Look Back   : {args.look_back}")
    print(f"Batch Size  : {args.bs}")
    print(f"Train Case  : {len(train_index_list)}")
    print(f"Val Case    : {len(val_index_list)}")
    train_start_time = time.perf_counter()
    for epoch in range(1, epoch_num + 1):
        print("----------------------------------------------")
        print(f"[Epoch {epoch}]")
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            inp = batch["inp"].to(device)
            spec = batch["spec"].to(device)
            gt = batch["gt"].to(device)

            pred, _ = model(inp, spec)

            loss = criterion(pred, gt)
            loss.backward()

            epoch_loss += loss.item() * inp.size(0)
            optimizer.step()

        epoch_loss = epoch_loss / len(train_dataloader)
        print(f"Train Loss: {epoch_loss}")

        with torch.no_grad():
            model.eval()
            epoch_test_error = 0
            for batch in tqdm(val_dataloader):
                inp = batch["inp"].to(device)
                spec = batch["spec"].to(device)
                gt = batch["gt"].to(device)

                pred, _ = model(inp, spec)

                test_error = torch.mean(torch.abs(gt - pred))
                epoch_test_error += test_error.item() * inp.size(0)

            epoch_test_error = epoch_test_error / len(val_dataloader)
            print(f"Val Loss: {epoch_test_error}")

        mlflow.log_metric(f"train loss", epoch_loss, step=epoch)
        mlflow.log_metric(f"val loss", epoch_test_error, step=epoch)

        result_path = os.path.join("..", "result")
        os.makedirs(result_path, exist_ok=True)
        early_stopping(epoch_test_error, model, epoch)
        if early_stopping.early_stop:
            break

    train_end_time = time.perf_counter()
    train_time = datetime.timedelta(seconds=(train_end_time - train_start_time))
    mlflow.log_metric(f"train time", train_end_time - train_start_time)

    print("----------------------------------------------")
    print(f"Train Time: {train_time}")

    with torch.no_grad():
        print(f"\n\nTest Start. Case Num: {len(test_index_list)}")
        print("----------------------------------------------")
        model.load_state_dict(torch.load(early_stopping.path))
        model.eval()
        enable_dropout(model)

        for test_index in tqdm(test_index_list):
            case_name = CaseNameDecision(test_index)

            output_feature_name = data["feature_name"][CFG.NUM_CONTROL_FEATURES + 1 :]
            output_feature_unit = data["feature_unit"][CFG.NUM_CONTROL_FEATURES + 1 :]
            inp_data = data["inp"][test_index]
            spec_data = data["spec"][test_index]
            gt_data = data["gt"][test_index]

            pred_output_data = []
            attn_output = []

            start_time = time.perf_counter()
            for j in range(10):
                attn_all = []
                scaling_input_data = inp_data[0].copy()
                scaling_spec_data = spec_data.copy()
                for i in range(scaling_input_data.shape[1]):  # input scaling
                    scaling_input_data[:, i] = (scaling_input_data[:, i] - mean_list[i]) / std_list[i]
                for i in range(scaling_spec_data.shape[1]):  # spec scaling
                    scaling_spec_data[:, i] = (scaling_spec_data[:, i] - mean_list[i]) / std_list[i]

                for i in range(scaling_spec_data.shape[0]):
                    input = torch.from_numpy(scaling_input_data[i : i + args.look_back].astype(np.float32)).clone().unsqueeze(0).to(device)
                    spec = torch.from_numpy(scaling_spec_data[i].astype(np.float32)).clone().unsqueeze(0).to(device)

                    scaling_pred_data, attn = model(input, spec)
                    scaling_pred_data = scaling_pred_data.detach().to("cpu").numpy().copy()[0]
                    new_scaling_input_data = np.append(scaling_spec_data[i], scaling_pred_data)[np.newaxis, :]
                    scaling_input_data = np.concatenate([scaling_input_data, new_scaling_input_data], axis=0)

                    if "BaseTransformer" in args.model:
                        if i == 0:
                            attn_all = attn.detach().to("cpu").numpy().copy()[np.newaxis]
                        else:
                            attn_all_ = attn.detach().to("cpu").numpy().copy()[np.newaxis]
                            attn_all = np.concatenate((attn_all, attn_all_), axis=0)

                if j == 0:
                    pred_output_data = scaling_input_data.copy()[np.newaxis, :, :]
                    attn_output = attn_all[np.newaxis]
                else:
                    pred_output_data_ = scaling_input_data.copy()[np.newaxis, :, :]
                    pred_output_data = np.concatenate((pred_output_data, pred_output_data_), axis=0)
                    attn_output = np.concatenate((attn_output, attn_all[np.newaxis]), axis=0)
            # print(pred_output_data.shape) # (10, 1199, 36)
            # print(attn_output.shape) # (10, 1199, 3, 8, 11, 11)
            end_time = time.perf_counter()
            predict_time_list.append(end_time - start_time)

            scaled_pred_data = pred_output_data[:, :, CFG.NUM_CONTROL_FEATURES :]

            for i in range(pred_output_data.shape[2]):  # undo scaling
                pred_output_data[:, :, i] = pred_output_data[:, :, i] * std_list[i] + mean_list[i]
            pred_data = pred_output_data[:, :, CFG.NUM_CONTROL_FEATURES :]

            gt_output_data = []
            for inp in inp_data[0]:
                gt_output_data.append(inp[CFG.NUM_CONTROL_FEATURES :])
            for gt in gt_data:
                gt_output_data.append(gt)

            scaling_gt_data = np.zeros(np.array(gt_output_data).shape)
            for i in range(np.array(gt_output_data).shape[1]):
                scaling_gt_data[:, i] = (np.array(gt_output_data)[:, i] - mean_list[i + CFG.NUM_CONTROL_FEATURES]) / std_list[i + CFG.NUM_CONTROL_FEATURES]

            evaluation(test_index, gt_output_data, pred_data, output_feature_name, case_name, CFG.NUM_FIXED_DATA)
            visualization(gt_output_data, pred_data, output_feature_name, output_feature_unit, case_name, result_path)
            visualization(scaling_gt_data, scaled_pred_data, output_feature_name, output_feature_unit, case_name, result_path, is_normalized=True)
            attention_visualization(args, attn_output, result_path, case_name) if "BaseTransformer" in args.model else None

    save_evaluation(result_path)
    mlflow.log_metric(f"predict time mean", np.mean(predict_time_list))
    mlflow.log_artifacts(local_dir=result_path, artifact_path="result")
    shutil.rmtree(result_path)
    mlflow.end_run()
    print("----------------------------------------------")
    print(f"Predict Time: {np.mean(predict_time_list)} [s]\n")
    print("Experiment End\n")


if __name__ == "__main__":
    main()
