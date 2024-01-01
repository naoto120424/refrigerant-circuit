import torch
from torch.utils.data import DataLoader
import numpy as np
import time, datetime, os
import mlflow, shutil, argparse

from tqdm import tqdm
from sklearn.model_selection import train_test_split

from utils.utils import CFG, criterion_list, predict_time_list, deviceDecision, seed_everything, modelDecision
from utils.dataloader import load_data, create_dataset
from utils.visualization import Eva, print_model_summary
from utils.earlystopping import EarlyStopping


def main():
    """parser"""
    parser = argparse.ArgumentParser(description="Mazda Refrigerant Circuit Project")

    parser.add_argument("--e_name", type=str, default="Mazda Refrigerant Circuit", help="experiment name")
    parser.add_argument("--model", type=str, default="BaseTransformer", help="model name")
    parser.add_argument("--debug", type=bool, default=False, help="debug")
    parser.add_argument("--seed", type=int, default=42, help="seed")

    parser.add_argument("--bs", type=int, default=32, help="batch size")
    parser.add_argument("--train_epochs", type=int, default=20, help="train epochs")
    parser.add_argument("--criterion", type=str, default="MSE", help="criterion name. MSE / L1")

    parser.add_argument("--patience", type=int, default=3, help="early stopping patience")
    parser.add_argument("--delta", type=float, default=0.0, help="early stopping delta")
    parser.add_argument("--learning_rate", type=float, default=1e-3, help="optimizer initial learning rate")  # add

    parser.add_argument("--in_len", type=int, default=12, help="input MTS length (T)")  # change look_back -> in_len
    parser.add_argument("--out_len", type=int, default=1, help="output MTS length (\tau)")  # add out_len

    parser.add_argument("--e_layers", type=int, default=3, help="num of encoder layers (N)")  # change depth -> e_layers
    parser.add_argument("--d_model", type=int, default=256, help="dimension of hidden states (d_model)")  # change dim -> d_model
    parser.add_argument("--dropout", type=float, default=0.1, help="dropout rate")
    parser.add_argument("--n_heads", type=int, default=8, help="num of heads of multi-head Attention")  # change heads -> n_heads
    parser.add_argument("--d_ff", type=int, default=512, help="dimension of MLP in transformer")  # change fc_dim -> d_ff

    parser.add_argument("--seg_len", type=int, default=3, help="(CrossFormer) segment length (L_seg)")
    parser.add_argument("--win_size", type=int, default=2, help="(CrossFormer) window size for segment merge")
    parser.add_argument("--factor", type=int, default=10, help="(CrossFormer) num of routers in Cross-Dimension Stage of TSA (c)")
    
    parser.add_argument("--trunk_layers", type=int, default=3, help="(DeepOx) num of Trunk Net Layers")
    parser.add_argument("--trunk_d_model", type=int, default=256, help="(DeepOx) Trunk Net dimension")

    args = parser.parse_args()

    """ mlflow """
    mlflow.set_tracking_uri(CFG.MLFLOW_PATH)
    mlflow.set_experiment(args.e_name)
    mlflow.start_run()
    mlflow.log_param("model", args.model)
    mlflow.log_param("debug", args.debug)
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("batch size", args.bs)
    mlflow.log_param("criterion", args.criterion)
    mlflow.log_param("patience", args.patience)
    mlflow.log_param("delta", args.delta)
    mlflow.log_param("in_len", args.in_len)
    mlflow.log_param("d_model", args.d_model)
    mlflow.log_param("e_layers", args.e_layers)
    mlflow.log_param("dropout", args.dropout)
    if ("BaseTransformer" in args.model) or ("Crossformer" in args.model) or ("DeepO" in args.model):
        mlflow.log_param("heads", args.n_heads)
        mlflow.log_param("d_ff", args.d_ff)
        if "Crossformer" in args.model:
            mlflow.log_param("seg_len", args.seg_len)
            mlflow.log_param("win_size", args.win_size)
            mlflow.log_param("factor", args.factor)
        if "DeepO" in args.model:
            mlflow.log_param("trunk layers", args.trunk_layers)
            mlflow.log_param("trunk dim", args.trunk_d_model)

    """Prepare"""
    seed_everything(seed=args.seed)
    data = load_data(CFG, in_len=args.in_len, debug=args.debug)
    eva = Eva(data)
    device = deviceDecision()

    if not args.debug:
        train_index_list, test_index_list = train_test_split(np.arange(len(data["inp"])), test_size=100)
        train_index_list, val_index_list = train_test_split(train_index_list, test_size=50)
    else:
        train_index_list, test_index_list = train_test_split(np.arange(25), test_size=5)
        train_index_list, val_index_list = train_test_split(train_index_list, test_size=10)

    train_dataset, mean_list, std_list = create_dataset(CFG, data, train_index_list, is_train=True)
    val_dataset, _, _ = create_dataset(CFG, data, val_index_list, is_train=False, mean_list=mean_list, std_list=std_list)

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True, num_workers=os.cpu_count())
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=True, num_workers=os.cpu_count())

    model = modelDecision(args, CFG)
    model.to(device)

    criterion = criterion_list[args.criterion]
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    early_stopping = EarlyStopping(patience=args.patience, delta=args.delta, verbose=True)
    epoch_num = args.train_epochs if not args.debug else 3

    # model = torch.compile(model)  # For PyTorch2.0

    """Train"""
    print_model_summary(args, device, len(train_index_list), len(val_index_list))
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
            timedata = batch["timedata"].to(device)
            
            if (args.model == "Transformer") or (args.model == "DeepOTransformer") or (args.model == "DeepONet"):
                pred, _ = model(inp, spec, timedata)  # Transformer, DeepOTransformer, DeepONet
            else:
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
                timedata = batch["timedata"].to(device)

                if (args.model == "Transformer") or (args.model == "DeepOTransformer") or (args.model == "DeepONet"):
                    pred, _ = model(inp, spec, timedata)
                else:
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

        for test_index in tqdm(test_index_list):  # 20230709 test_index_list -> train_index_list (訓練データでの予測でもうまくいかないのかを確認する)
            case_name = f"case{str(test_index+1).zfill(4)}"

            output_feature_name = data["feature_name"][CFG.NUM_CONTROL_FEATURES + 1 :]
            output_feature_unit = data["feature_unit"][CFG.NUM_CONTROL_FEATURES + 1 :]
            inp_data = data["inp"][test_index]
            spec_data = data["spec"][test_index]
            gt_data = data["gt"][test_index]
            time_data = data["timedata"][test_index]

            scaling_input_data = inp_data[0].copy()
            scaling_spec_data = spec_data.copy()
            scaling_gt_data = gt_data.copy()
            scaling_time_data = time_data.copy()
            for i in range(scaling_input_data.shape[1]):  # input scaling
                scaling_input_data[:, i] = (scaling_input_data[:, i] - mean_list[i+1]) / std_list[i+1]
            for i in range(scaling_spec_data.shape[1]):  # spec scaling
                scaling_spec_data[:, i] = (scaling_spec_data[:, i] - mean_list[i+1]) / std_list[i+1]
            for i in range(scaling_gt_data.shape[1]):  # ground truth scaling
                scaling_gt_data[:, i] = (scaling_gt_data[:, i] - mean_list[i+CFG.NUM_CONTROL_FEATURES+1]) / std_list[i+CFG.NUM_CONTROL_FEATURES+1]
            for i in range(scaling_time_data.shape[1]):
                scaling_time_data[:, i] = (scaling_time_data[:, i] - mean_list[0]) / std_list[0]

            attn_all = []

            start_time = time.perf_counter()

            for i in range(scaling_spec_data.shape[0]):
                input = torch.from_numpy(scaling_input_data[i : i + args.in_len].astype(np.float32)).clone().unsqueeze(0).to(device)
                spec = torch.from_numpy(scaling_spec_data[i].astype(np.float32)).clone().unsqueeze(0).to(device)
                gt = torch.from_numpy(scaling_gt_data[i].astype(np.float32)).clone().unsqueeze(0).to(device)
                timedata = torch.from_numpy(scaling_time_data[i].astype(np.float32)).clone().unsqueeze(0).to(device)

                if (args.model == "Transformer") or (args.model == "DeepOTransformer") or (args.model == "DeepONet"):
                    scaling_pred_data, attn = model(input, spec, timedata)  # test pred here
                else:
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
                pred_output_data[:, i] = pred_output_data[:, i] * std_list[i+1] + mean_list[i+1]
            pred_data = pred_output_data[:, CFG.NUM_CONTROL_FEATURES :]
            scaling_pred_data = scaling_input_data[:, CFG.NUM_CONTROL_FEATURES :]

            gt_output_data = []  # ground truth data for visualization
            for inp in inp_data[0]:
                gt_output_data.append(inp[CFG.NUM_CONTROL_FEATURES :])
            for gt in gt_data:
                gt_output_data.append(gt)

            scaling_gt_data = np.zeros(np.array(gt_output_data).shape)
            for i in range(np.array(gt_output_data).shape[1]):  # scaling ground truth data for visualization
                scaling_gt_data[:, i] = (np.array(gt_output_data)[:, i] - mean_list[i+CFG.NUM_CONTROL_FEATURES+1]) / std_list[i+CFG.NUM_CONTROL_FEATURES+1]

            eva.evaluation(args.in_len, gt_output_data, pred_data, case_name)
            eva.visualization(gt_output_data, pred_data, case_name, CFG.RESULT_PATH)
            eva.visualization(scaling_gt_data, scaling_pred_data, case_name, CFG.RESULT_PATH, is_normalized=True)
            eva.attention_visualization(args, attn_all, CFG.RESULT_PATH, case_name) if "BaseTransformer" in args.model else None

    eva.save_evaluation(CFG.RESULT_PATH)
    mlflow.log_metric(f"predict time mean", np.mean(predict_time_list))
    mlflow.log_artifacts(local_dir=CFG.RESULT_PATH, artifact_path="result")
    shutil.rmtree(CFG.RESULT_PATH)
    mlflow.end_run()
    print("----------------------------------------------")
    print(f"Predict Time: {np.mean(predict_time_list)} [s]\n")
    print("Experiment End")


if __name__ == "__main__":
    if os.path.isdir(CFG.RESULT_PATH):
        shutil.rmtree(CFG.RESULT_PATH)
    main()
