import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import random
import shutil
import numpy as np
from dataloader import *
from sklearn.model_selection import train_test_split
import mlflow

# 毎回同じ乱数が出るようにs
def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

class LSTMClassifier(nn.Module):
    def __init__(self, input_dim=39, num_hidden_units=256, spec_dim=9, output_dim=30):
        super(LSTMClassifier, self).__init__()
        self.input_dim = input_dim
        self.num_hidden_units = num_hidden_units
        self.lstm = nn.LSTM(input_size=input_dim, 
                            hidden_size=num_hidden_units,
                            num_layers=2,
                            dropout=0.2,
                            batch_first=True
                            )
        self.spec_dense = nn.Linear(spec_dim, num_hidden_units)
        self.predicter = nn.Sequential(
            nn.Linear(num_hidden_units*2, num_hidden_units),
            nn.Linear(num_hidden_units, output_dim)
        )

    def forward(self, x, spec, h=None):
        hidden1, _ = self.lstm(x, h)
        hidden2 = self.spec_dense(spec)
        y = self.predicter(torch.cat([hidden1[:, -1, :], hidden2], dim=1)) # 最後のセルだけを取り出している
        return y

def deviceChecker():
    if torch.backends.mps.is_available():
        return "mps"
    elif torch.cuda.is_available():
        return "cuda"
    else:
        return "cpu"


def main():
    seed = 42
    epoch_num = 500
    batch_size = 64
    look_back = 40
    
    mlflow.set_tracking_uri('/home/sasasho/researchment/refrigerant-circuit/mlfow_experiment')
    mlflow.set_experiment('Mazda Refrigerant Circuit Turtrial')
    mlflow.start_run()
    mlflow.log_param("seed", seed)
    mlflow.log_param("batch size", batch_size)
    mlflow.log_param("look back size", look_back)

    seed_everything(seed=seed)
    device = deviceChecker()
    print('device: ' + device)

    data = load_data(look_back)

    data_size = len(data['inp'])
    print(f'data size: {data_size}')

    train_index_list, test_index_list = train_test_split(np.arange(data_size), test_size=10)
    train_index_list, val_index_list = train_test_split(train_index_list,test_size=10)

    train_dataset, mean_list, std_list = create_dataset(data, train_index_list, is_train=True)
    val_dataset, _, _ = create_dataset(data, val_index_list, is_train=False, mean_list=mean_list, std_list=std_list)
    # test_dataset, _, _ = create_dataset(data, train_index_list, is_train=False, mean_list=mean_list, std_list=std_list)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    # Prepare for training
    model = LSTMClassifier().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    best_loss = 100.0
    for epoch in range(epoch_num):
        print('-----------')
        print(f'Epoch {epoch+1}/{epoch_num}')
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            inputs = batch['inp'].to(device)    # shape(batch_size, 20, 39)
            spec = batch['spec'].to(device)     # shape(batch_size, 9)
            label = batch['gt'].to(device)      # shape(batch_size, 30)

            outputs = model(inputs, spec)
            loss = criterion(outputs, label)
            loss.backward()
            epoch_loss += loss.item() * inputs.size(0)
            optimizer.step()
        epoch_loss = epoch_loss / len(train_dataloader)
        print(f'train Loss: {epoch_loss}') 
        mlflow.log_metric(f'train loss', epoch_loss, step=epoch)

        with torch.no_grad():
            model.eval()
            epoch_test_error = 0
            for batch in tqdm(val_dataloader):
                inputs = batch['inp'].to(device)
                spec = batch['spec'].to(device)
                label = batch['gt'].to(device)
                
                output = model(inputs, spec)

                test_error = torch.mean(torch.abs(label - output))
                
                epoch_test_error += test_error.item() * inputs.size(0)
            
            epoch_test_error = epoch_test_error / len(val_dataloader)
            print(f'val Loss: {epoch_test_error}')
            mlflow.log_metric(f'val loss', epoch_test_error, step=epoch)
        
        # save best model
        if epoch_test_error < best_loss:
            best_epoch_num = epoch + 1
            best_loss = epoch_test_error
            print('This is the best model.')
            print('-----SAVE-----')
            model_path = 'best_model.pth'
            torch.save(model.state_dict(), model_path)
    mlflow.log_metric("best epoch num", best_epoch_num)
    
    # test
    result_path = os.path.join('..', 'result')
    with torch.no_grad():
        model_path = 'best_model.pth'
        model.load_state_dict(torch.load(model_path))
        model.eval()
        os.makedirs(result_path, exist_ok=True)
        for test_index in test_index_list:
            case_name = f'case{str(test_index+1).zfill(4)}'
            print(case_name)
            case_path = os.path.join(result_path, case_name)
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
    
    mlflow.log_artifacts(local_dir=result_path, artifact_path='result')
    # mlflow上に保存するので削除する
    shutil.rmtree(result_path)
    mlflow.end_run()


if __name__ == '__main__':
    main()
