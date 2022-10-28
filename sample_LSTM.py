import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import numpy as np
import random
import numpy as np
import pandas as pd
from utils.dataloader import *
from sklearn.model_selection import train_test_split

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
    seed_everything()
    epoch_num = 10
    device = deviceChecker()
    print('device: ' + device)

    data = load_data(look_back=20)
    data_size = len(data['inp'])
    print(f'data size: {data_size}')

    train_index_list, test_index_list = train_test_split(np.arange(data_size), test_size=10)
    train_index_list, val_index_list = train_test_split(train_index_list,test_size=10)

    train_dataset, mean_list, std_list = create_dataset(data, train_index_list, is_train=True)
    val_dataset, _, _ = create_dataset(data, val_index_list, is_train=False, mean_list=mean_list, std_list=std_list)
    # test_dataset, _, _ = create_dataset(data, train_index_list, is_train=False, mean_list=mean_list, std_list=std_list)

    train_dataloader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=64, shuffle=True)
    # test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)
    # 以下略

    # Prepare for training
    model = LSTMClassifier().to(device)
    criterion = nn.L1Loss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)

    for epoch in range(epoch_num):
        print('-----------')
        print(f'Epoch {epoch+1}/{epoch_num}')
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader):
            optimizer.zero_grad()
            inputs = batch['inp'].to(device)
            spec = batch['spec'].to(device)
            label = batch['gt'].to(device)

            outputs = model(inputs, spec)
            loss = criterion(outputs, label)
            loss.backward()
            epoch_loss += loss.item() * inputs.size(0)
            optimizer.step()
        epoch_loss = epoch_loss / len(train_dataloader)
        print(f'train Loss: {epoch_loss}') 

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


if __name__ == '__main__':
    main()
