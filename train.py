import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt
import numpy as np
import mlflow
import shutil
from utils.dataloader import *
from utils.utiles import *
from sklearn.model_selection import train_test_split

def main():
    seed = 42
    epoch_num = 10
    batch_size = 20
    look_back = 20

    mlflow.set_tracking_uri('../mazda/mlflow_experiment')
    mlflow.set_experiment('Mazda Refrigerant Circuit Turtrial')
    mlflow.start_run()
    mlflow.log_param("seed", seed)
    mlflow.log_param("batch size", batch_size)
    mlflow.log_param("look back size", look_back)

    seed_everything(seed=seed)

    device = deviceChecker()
    print(f'using {device} device\n')

    data = load_data(look_back=look_back)

    train_index_list, test_index_list = train_test_split(np.arange(100), test_size=10)
    train_index_list, val_index_list = train_test_split(train_index_list,test_size=10)

    print("\ncreating dataset and normalisation now...")
    train_dataset, mean_list, std_list = create_dataset(data, train_index_list, is_train=True)
    val_dataset, _, _ = create_dataset(data, val_index_list, is_train=False, mean_list=mean_list, std_list=std_list)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    model = LSTMClassifier().to(device)
    criterion = nn.MSELoss()
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

        result_path = os.path.join('..', 'mazda', 'result')
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
            # print(case_name)
            case_path = os.path.join(result_path, 'img', case_name)
            os.makedirs(case_path, exist_ok=True)
            inp_data = data['inp'][test_index]
            spec_data = data['spec'][test_index]
            gt_data = data['gt'][test_index]
            # print("inp_data.shape", inp_data.shape)
            # print("spec_data.shape", spec_data.shape)
            # print("gt_data.shape", gt_data.shape)

            # 標準化処理
            scaling_input_data = inp_data[0].copy()
            scaling_spec_data = spec_data.copy()
            for i in range(scaling_input_data.shape[1]):
                scaling_input_data[:,i] = (scaling_input_data[:,i]-mean_list[i])/std_list[i]
            for i in range(scaling_spec_data.shape[1]):
                scaling_spec_data[:,i] = (scaling_spec_data[:,i]-mean_list[i])/std_list[i]
            # for i in range(scaling_gt_data.shape[1]):
            #     scaling_gt_data[:,i] = (scaling_gt_data[:,i]-mean_list[i+scaling_spec_data.shape[1]])/std_list[i+scaling_spec_data.shape[1]]

            for i in range(scaling_spec_data.shape[0]):
                input = torch.from_numpy(scaling_input_data[i:i+look_back].astype(np.float32)).clone().unsqueeze(0).to(device)
                spec = torch.from_numpy(scaling_spec_data[i].astype(np.float32)).clone().unsqueeze(0).to(device)
                # print(input.shape)
                # print(spec.shape)
                scaling_pred_data = model(input, spec).detach().to('cpu').numpy().copy()[0]
                # print("scaling_pred_data", scaling_pred_data.shape)
                new_scaling_input_data = np.append(scaling_spec_data[i], scaling_pred_data)[np.newaxis, :]
                # print("new_scaling_data", new_scaling_input_data.shape)
                scaling_input_data = np.concatenate([scaling_input_data, new_scaling_input_data], axis=0)
                # print("scaling_input_data", scaling_input_data.shape)

            # 標準化をもとに戻す処理
            pred_data = scaling_input_data.copy()
            for i in range(scaling_input_data.shape[1]):
                pred_data[:,i] = pred_data[:,i] * std_list[i] + mean_list[i]
            pred_output_data = pred_data[:,9:]

            # gtの出力を作成
            output_feature_name = data['feature_name'][10:]
            output_feature_unit = data['feature_unit'][10:]
            gt_output_data = []
            for inp in inp_data[0]:
                gt_output_data.append(inp[9:])
            for gt in gt_data:
                gt_output_data.append(gt)

            # 波形の出力
            for i in range(len(output_feature_name)):
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
    # mlflow上に保存するので削除する
    shutil.rmtree(result_path)
    mlflow.end_run() 

if __name__ == '__main__':
    main()
