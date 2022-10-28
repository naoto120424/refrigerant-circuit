import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import numpy as np
from utils.dataloader import *
from utils.utiles import *
from sklearn.model_selection import train_test_split


def main():
    seed = 42
    epoch_num = 2
    batch_size = 20
    look_back = 20

    seed_everything(seed)

    device = deviceChecker()
    print(f'using {device} device\n')

    data = load_data(look_back=20)

    train_index_list, test_index_list = train_test_split(np.arange(100), test_size=10)
    train_index_list, val_index_list = train_test_split(train_index_list,test_size=10)

    print("\ncreating dataset and normalisation now...")
    train_dataset, mean_list, std_list = create_dataset(data, train_index_list, is_train=True)
    val_dataset, _, _ = create_dataset(data, val_index_list, is_train=False, mean_list=mean_list, std_list=std_list)
    test_dataset, _, _ = create_dataset(data, test_index_list, is_train=False, mean_list=mean_list, std_list=std_list)

    train_dataloader = DataLoader(train_dataset, batch_size=20, shuffle=True)
    val_dataloader = DataLoader(val_dataset, batch_size=20, shuffle=False)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=False)

    model = LSTMClassifier().to(device)
    criterion = nn.MSELoss()
    optimizer = torch.optim.SGD(model.parameters(), lr=1e-3)
    

    best_loss = 100.0
    best_epoch_num = 0
    print('\ntrain start')
    for epoch in range(1, epoch_num + 1):
        print('------------------------------------')
        print(f'Epoch {epoch}/{epoch_num}')
        model.train()
        epoch_loss = 0.0
        for batch in tqdm(train_dataloader, position=0, dynamic_ncols=True):
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

        with torch.no_grad():
            model.eval()
            epoch_test_error = 0
            for batch in tqdm(val_dataloader, position=0, dynamic_ncols=True):
                inp = batch['inp'].to(device)
                spec = batch['spec'].to(device)
                gt = batch['gt'].to(device)
                
                pred = model(inp, spec)

                test_error = torch.mean(torch.abs(gt - pred))
                
                epoch_test_error += test_error.item() * inp.size(0)
            
            epoch_test_error = epoch_test_error / len(val_dataloader)
            print(f'train Loss: {epoch_loss}, val Loss: {epoch_test_error}')

        if(epoch_test_error < best_loss):
            best_epoch_num = epoch
            best_loss = epoch_test_error
            print('This is the best model. Save to "best_model.pth".')
            model_path = os.path.join('weight', 'best_model.pth')
            torch.save(model.state_dict(), model_path)


    with open(os.path.join('weight', 'best_model_num.txt'), 'w') as f:
        f.write(f'best_epoch_num:{best_epoch_num}')

    print('------------------------------------\n')
    model_path = os.path.join('weight', 'best_model.pth')
    loss_hr_rate_max_list = []
    loss_integrated_hr_list = []
    loss_x50_ca_list = []
    loss_distance_list = []
    output_csv_list = []
    output_csv_columns = ['case'] + [i for i in range(30)]
    with torch.no_grad():
        print('------------------------------------')
        print("test start")
        model.load_state_dict(torch.load(model_path))
        model.eval()
        for test_index in tqdm(test_index_list, position=0, dynamic_ncols=True):
            case = data['data_name'][test_index]
            case_num = int(case[4:].lstrip('0'))
            save_path = os.path.join('figure', case)
            os.makedirs(save_path, exist_ok=True)
            state_quantity = data['inp'][test_index]

            state_quantity = (state_quantity - mean_list) / std_list
            state_quantity = state_quantity.astype(np.float32)
            state_quantity = torch.from_numpy(state_quantity).clone().to(device)
            # state_quantity = state_quantity.unsqueeze(0).to(device)
            # print(state_quantity.shape)

            spec = data['spec'][test_index]
            mean_spec, std_spec = mean_list[:9], std_list[:9]
            spec = (spec - mean_spec) / std_spec
            spec = spec.astype(np.float32)
            spec = torch.from_numpy(spec).clone().to(device)
            # spec = spec.unsqueeze(0).to(device)

            pred = model(state_quantity, spec).detach().to('cpu').numpy().copy()[0]
            gt = data['gt'][test_index]

            output_csv_list.append([case] + pred.tolist())
            pd_output_csv = pd.DataFrame(output_csv_list, columns=output_csv_columns).set_index('case').T
            pd_output_csv.index.name = 'time'
            # pd_output_csv.to_csv(os.path.join(experiment_path, 'csv', f'output_fold{fold_index+1}.csv'))
            
            #評価指標に基づきコード作成

if __name__ == '__main__':
    main()
