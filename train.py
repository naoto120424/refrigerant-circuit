import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import shutil
import argparse
from utils.dataloader import *
from utils.utiles import *
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error
from model.lstm import LSTMClassifier
from model.base_transformer import BaseTransformer

def main():
    parser=argparse.ArgumentParser(description="Mazda Refrigerant Circuit Project")
    parser.add_argument('--e_name',type=str,default='Mazda Refrigerant Circuit Turtrial')
    parser.add_argument('--seed',type=int,default=42)
    parser.add_argument('--bs',type=int,default=128)
    parser.add_argument('--epoch',type=int,default=500)
    parser.add_argument('--look_back',type=int,default=20)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--model', type=str, default="BaseTransformer")
    parser.add_argument('--criterion', type=str, default="mse")
    args=parser.parse_args()

    if args.debug:
        epoch_num = 3
    else:
        epoch_num = args.epoch

    mlflow.set_tracking_uri('../mlflow_experiment')
    mlflow.set_experiment(args.e_name)
    mlflow.start_run()
    mlflow.log_param("seed", args.seed)
    mlflow.log_param("epoch num", epoch_num)
    mlflow.log_param("batch size", args.bs)
    mlflow.log_param("look back size", args.look_back)
    mlflow.log_param("debug", args.debug)
    mlflow.log_param("model", args.model)
    mlflow.log_param("criterion", args.criterion)

    seed_everything(seed=args.seed)

    device = deviceChecker()
    print(f'device: {device}')

    data = load_data(look_back=args.look_back)

    train_index_list, test_index_list = train_test_split(np.arange(100), test_size=10)
    train_index_list, val_index_list = train_test_split(train_index_list,test_size=10)

    print("creating dataset and normalisation...")
    train_dataset, mean_list, std_list = create_dataset(data, train_index_list, is_train=True)
    val_dataset, _, _ = create_dataset(data, val_index_list, is_train=False, mean_list=mean_list, std_list=std_list)

    train_dataloader = DataLoader(train_dataset, batch_size=args.bs, shuffle=False)
    val_dataloader = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

    if args.model in model_list:
        model = model_list[args.model]
    elif args.model == "BaseTransformer":
        model = BaseTransformer(look_back=args.look_back)
    else:
        print("unknown model name")
        return

    model.to(device)
    
    if args.criterion in criterion_list:
        criterion = criterion_list[args.criterion]
    else:
        print("unknown criterion name")
        return

    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    
    best_loss = 100.0
    print('\ntrain start')
    for epoch in range(1, epoch_num + 1):
        print('------------------------------------')
        print(f'Epoch {epoch}/{epoch_num}')
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            inp = batch['inp'].to(device)
            spec = batch['spec'].to(device)
            gt = batch['gt'].to(device)

            pred = model(inp, spec)
            loss = criterion(pred, gt)
            loss.backward()
            epoch_loss += loss.item() * inp.size(0)
            optimizer.step()
        epoch_loss = epoch_loss / len(train_dataloader)
        mlflow.log_metric(f'train loss', epoch_loss, step=epoch)

        with torch.no_grad():
            model.eval()
            epoch_test_error = 0
            for batch in tqdm(val_dataloader):
                inp = batch['inp'].to(device)
                spec = batch['spec'].to(device)
                gt = batch['gt'].to(device)
                
                pred = model(inp, spec)
                test_error = torch.mean(torch.abs(gt - pred))
                epoch_test_error += test_error.item() * inp.size(0)
            
            epoch_test_error = epoch_test_error / len(val_dataloader)
            print(f'train Loss: {epoch_loss}, val Loss: {epoch_test_error}')
            mlflow.log_metric(f'val loss', epoch_test_error, step=epoch)

        result_path = os.path.join('..', 'result')
        os.makedirs(result_path, exist_ok=True)

        if(epoch_test_error < best_loss):
            best_epoch_num = epoch
            best_loss = epoch_test_error
            print('This is the best model. Save to "best_model.pth".')
            model_path = os.path.join(result_path, 'best_model.pth')
            torch.save(model.state_dict(), model_path)
    mlflow.log_metric(f'best epoch num', best_epoch_num)

    print('------------------------------------\n')

    with torch.no_grad():
        print("test start")
        print('------------------------------------')
        model.load_state_dict(torch.load(model_path))
        model.eval()
        for test_index in tqdm(test_index_list):
            case_name = f'case{str(test_index+1).zfill(4)}'
            case_path = os.path.join(result_path, 'img', case_name)
            os.makedirs(case_path, exist_ok=True)
            inp_data = data['inp'][test_index]
            spec_data = data['spec'][test_index]
            gt_data = data['gt'][test_index]

            scaling_input_data = inp_data[0].copy()
            scaling_spec_data = spec_data.copy()
            for i in range(scaling_input_data.shape[1]):
                scaling_input_data[:,i] = (scaling_input_data[:,i]-mean_list[i])/std_list[i]
            for i in range(scaling_spec_data.shape[1]):
                scaling_spec_data[:,i] = (scaling_spec_data[:,i]-mean_list[i])/std_list[i]

            for i in range(scaling_spec_data.shape[0]):
                input = torch.from_numpy(scaling_input_data[i:i+args.look_back].astype(np.float32)).clone().unsqueeze(0).to(device)
                spec = torch.from_numpy(scaling_spec_data[i].astype(np.float32)).clone().unsqueeze(0).to(device)
                scaling_pred_data = model(input, spec).detach().to('cpu').numpy().copy()[0]
                new_scaling_input_data = np.append(scaling_spec_data[i], scaling_pred_data)[np.newaxis, :]
                scaling_input_data = np.concatenate([scaling_input_data, new_scaling_input_data], axis=0)

            pred_data = scaling_input_data.copy()
            for i in range(scaling_input_data.shape[1]):
                pred_data[:,i] = pred_data[:,i] * std_list[i] + mean_list[i]
            pred_output_data = pred_data[:,9:]

            output_feature_name = data['feature_name'][10:]
            output_feature_unit = data['feature_unit'][10:]
            gt_output_data = []
            for inp in inp_data[0]:
                gt_output_data.append(inp[9:])
            for gt in gt_data:
                gt_output_data.append(gt)

            for i in range(len(output_feature_name)):
                if output_feature_name[i] in target_kW:
                    mse = mean_squared_error(np.array(gt_output_data)[:,i], np.array(pred_output_data)[:,i])
                    mae = mean_absolute_error(np.array(gt_output_data)[:,i], np.array(pred_output_data)[:,i])
                    fde = abs(gt_output_data[-1][i] - pred_output_data[-1][i])
                    mlflow.log_metric(f'mse_' + output_feature_name[i] + f'_' + case_name, mse)
                    mlflow.log_metric(f'mae_' + output_feature_name[i] + f'_' + case_name, mae)
                    mlflow.log_metric(f'fde_' + output_feature_name[i] + f'_' + case_name, fde)
                fig = plt.figure()
                ax = fig.add_subplot(1, 1, 1)
                ax.plot(np.array(gt_output_data)[:,i], color='#e46409', label='gt')
                ax.plot(np.array(pred_output_data)[:,i], color='b', label='pred')
                ax.set_title(case_name)
                ax.set_xlabel('Time[s]')
                ax.set_ylabel(f'{output_feature_name[i]}[{output_feature_unit[i]}]')
                ax.legend(loc='best')
                plt.savefig(os.path.join(case_path, f'{output_feature_name[i]}.png'))
                plt.close()
    
    mlflow.log_artifacts(local_dir=result_path, artifact_path='result')
    shutil.rmtree(result_path)
    mlflow.end_run()
    print('------------------------------------')
    print("experiment end")

if __name__ == '__main__':
    main()
