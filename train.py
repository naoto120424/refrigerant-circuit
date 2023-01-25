import torch
from torch.utils.data import DataLoader
import numpy as np
import time, datetime, os
import mlflow, shutil, argparse

from utils.utils import *
from utils.dataloader import *
from utils.visualization import *
from utils.earlystopping import EarlyStopping
from sklearn.model_selection import train_test_split


def main():
    """parser"""
    parser = argparse.ArgumentParser(description="Mazda Refrigerant Circuit Project")
    parser.add_argument("--e_name", type=str, default="Mazda Refrigerant Circuit Tutorial")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--bs", type=int, default=16)
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

    """mlflow"""
    mlflow.set_tracking_uri("../mlflow_experiment")
    mlflow.set_experiment(args.e_name)
    mlflow.start_run()
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("batch size", args.bs)
    mlflow.log_param("look back", args.look_back)
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

    """Prepare"""
    seed_everything(seed=args.seed)
    data = load_data(CFG, look_back=args.look_back)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    if not args.debug:
        train_index_list, test_index_list = train_test_split(np.arange(len(data["inp"])), test_size=200)
        train_index_list, val_index_list = train_test_split(train_index_list, test_size=100)
    else:
        train_index_list, test_index_list = train_test_split(np.arange(100), test_size=10)
        train_index_list, val_index_list = train_test_split(train_index_list, test_size=10)

    train_dataset, mean_list, std_list = create_dataset(CFG, data, train_index_list, is_train=True)
    val_dataset, _, _ = create_dataset(CFG, data, val_index_list, is_train=False, mean_list=mean_list, std_list=std_list)

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=os.cpu_count())
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True, num_workers=os.cpu_count())

    model = modelDecision(args, CFG)
    criterion = criterion_list[args.criterion]

    model.to(device)
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, verbose=True)
    epoch_num = CFG.MAX_EPOCH if not args.debug else 3

    """Train"""
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

            # pred, _ = model(inp, spec, gt)  # Transformer
            pred, _ = model(inp, spec)  # train pred here

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

                # pred, _ = model(inp, spec, gt)  # Transformer
                pred, _ = model(inp, spec)  # validation pred here

                test_error = torch.mean(torch.abs(gt - pred))
                epoch_test_error += test_error.item() * inp.size(0)

            epoch_test_error = epoch_test_error / len(val_dataloader)
            print(f"Val Loss: {epoch_test_error}")

        mlflow.log_metric(f"train loss", epoch_loss, step=epoch)
        mlflow.log_metric(f"val loss", epoch_test_error, step=epoch)

        os.makedirs(CFG.RESULT_PATH, exist_ok=True)
        early_stopping(epoch_test_error, model, epoch)
        if early_stopping.early_stop:
            break

    train_end_time = time.perf_counter()
    train_time = datetime.timedelta(seconds=(train_end_time - train_start_time))
    mlflow.log_metric(f"train time", train_end_time - train_start_time)

    print("----------------------------------------------")
    print(f"Train Time: {train_time}")

    """Test"""
    with torch.no_grad():
        print(f"\n\nTest Start. Case Num: {len(test_index_list)}")
        print("----------------------------------------------")
        model.load_state_dict(torch.load(early_stopping.path))
        model.eval()

        for test_index in tqdm(test_index_list):
            case_name = f"case{str(test_index+1).zfill(4)}"

            output_feature_name = data["feature_name"][CFG.NUM_CONTROL_FEATURES + 1 :]
            output_feature_unit = data["feature_unit"][CFG.NUM_CONTROL_FEATURES + 1 :]
            inp_data = data["inp"][test_index]
            spec_data = data["spec"][test_index]
            gt_data = data["gt"][test_index]

            scaling_input_data = inp_data[0].copy()
            scaling_spec_data = spec_data.copy()
            scaling_gt_data = gt_data.copy()
            for i in range(scaling_input_data.shape[1]):  # input scaling
                scaling_input_data[:, i] = (scaling_input_data[:, i] - mean_list[i]) / std_list[i]
            for i in range(scaling_spec_data.shape[1]):  # spec scaling
                scaling_spec_data[:, i] = (scaling_spec_data[:, i] - mean_list[i]) / std_list[i]
            for i in range(scaling_gt_data.shape[1]):  # ground truth scaling
                scaling_gt_data[:, i] = (scaling_gt_data[:, i] - mean_list[i + CFG.NUM_CONTROL_FEATURES]) / std_list[i + CFG.NUM_CONTROL_FEATURES]

            attn_all = []

            start_time = time.perf_counter()
            for i in range(scaling_spec_data.shape[0]):
                input = torch.from_numpy(scaling_input_data[i : i + args.look_back].astype(np.float32)).clone().unsqueeze(0).to(device)
                spec = torch.from_numpy(scaling_spec_data[i].astype(np.float32)).clone().unsqueeze(0).to(device)
                gt = torch.from_numpy(scaling_gt_data[i].astype(np.float32)).clone().unsqueeze(0).to(device)

                scaling_pred_data, attn = model(input, spec)  # test pred here
                scaling_pred_data = scaling_pred_data.detach().to("cpu").numpy().copy()[0]
                new_scaling_input_data = np.append(scaling_spec_data[i], scaling_pred_data)[np.newaxis, :]
                scaling_input_data = np.concatenate([scaling_input_data, new_scaling_input_data], axis=0)

                if "BaseTransformer" in args.model:
                    attn = attn.detach().to("cpu").numpy().copy()
                    attn_all.append(attn)
            end_time = time.perf_counter()
            predict_time_list.append(end_time - start_time)

            pred_output_data = scaling_input_data.copy()
            for i in range(scaling_input_data.shape[1]):  # undo scaling
                pred_output_data[:, i] = pred_output_data[:, i] * std_list[i] + mean_list[i]
            pred_data = pred_output_data[:, CFG.NUM_CONTROL_FEATURES :]
            scaling_pred_data = scaling_input_data[:, CFG.NUM_CONTROL_FEATURES :]

            gt_output_data = []  # ground truth data for visualization
            for inp in inp_data[0]:
                gt_output_data.append(inp[CFG.NUM_CONTROL_FEATURES :])
            for gt in gt_data:
                gt_output_data.append(gt)

            scaling_gt_data = np.zeros(np.array(gt_output_data).shape)
            for i in range(np.array(gt_output_data).shape[1]):  # scaling ground truth data for visualization
                scaling_gt_data[:, i] = (np.array(gt_output_data)[:, i] - mean_list[i + CFG.NUM_CONTROL_FEATURES]) / std_list[i + CFG.NUM_CONTROL_FEATURES]

            evaluation(gt_output_data, pred_data, output_feature_name, case_name)
            visualization(gt_output_data, pred_data, output_feature_name, output_feature_unit, case_name, CFG.RESULT_PATH)
            visualization(scaling_gt_data, scaling_pred_data, output_feature_name, output_feature_unit, case_name, CFG.RESULT_PATH, is_normalized=True)
            attention_visualization(args, attn_all, CFG.RESULT_PATH, case_name) if "BaseTransformer" in args.model else None

    save_evaluation(CFG.RESULT_PATH)
    mlflow.log_metric(f"predict time mean", np.mean(predict_time_list))
    mlflow.log_artifacts(local_dir=CFG.RESULT_PATH, artifact_path="result")
    shutil.rmtree(CFG.RESULT_PATH)
    mlflow.end_run()
    print("----------------------------------------------")
    print(f"Predict Time: {np.mean(predict_time_list)} [s]\n")
    print("Experiment End")


if __name__ == "__main__":
    main()
