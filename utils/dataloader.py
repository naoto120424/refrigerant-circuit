import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
from einops import rearrange

#データ間引き関数
def decimate(data):
    dt=1
    new_data=[]
    pick_time=dt
    for i in range(len(data)):
        if data[i,0]>pick_time:
            x=data[i,0],data[i-1,0]
            y=data[i,:],data[i-1,:]
            a,b=np.polyfit(x,y,1) # n秒とn+0.1秒間の線形の近似式の傾きa、切片bを求める
            new_data.append([a*pick_time+b]) # n秒のデータを近似式から求める
            pick_time+=dt
    new_data=np.array(new_data)
    new_data=np.reshape(new_data,(new_data.shape[0],new_data.shape[2]))
    return new_data

# 全データをロードする関数
def load_data(look_back=20):
    print('Load Data')
    print('----------------------------------------------')
    data_path='../dataset/' 
    csv_files = os.listdir(data_path)
    csv_files.sort()

    data = {}
    Xdata = []
    Specdata = []
    Ydata = []

    for file in tqdm(csv_files):
        csv_data = pd.read_csv(os.path.join(data_path, file), skiprows=1).values
        single_data = decimate(csv_data)[:, 1:]
        # single_data = csv_data[:, 1:]
        spec_data = single_data[:, :9]
        output_data = single_data[:, 9:]
        input_time_list = []
        spec_list = []
        gt_list = []
        for t in range(single_data.shape[0]-look_back):  # data.shape[0]=1199
            input_time_list.append(single_data[t:t+look_back])
            spec_list.append(spec_data[t+look_back])
            gt_list.append(output_data[t+look_back])
        Xdata.append(np.array(input_time_list))
        Specdata.append(np.array(spec_list))
        Ydata.append(np.array(gt_list))
    data['inp'] = np.array(Xdata)     # shape(case_num, 1179, 20, 39)
    data['spec'] = np.array(Specdata) # shape(case_num, 1179, 9)
    data['gt'] = np.array(Ydata)      # shape(case_num, 1179, 30)
    # print(data['inp'].shape)
    # print(data['spec'].shape)
    # print(data['gt'].shape)
    data['feature_name'] = list(map(lambda x: x.split('.')[0], pd.read_csv(
        os.path.join(data_path, csv_files[0]), skiprows=0).columns.values))
    data['feature_unit'] = list(map(lambda x: x.split('.')[0], pd.read_csv(
        os.path.join(data_path, csv_files[0]), skiprows=1).columns.values))
    print('----------------------------------------------')
    return data

# 訓練データから標準化用の平均と標準偏差を求める関数
def find_meanstd(train_index_list):
    data_path='../dataset/'  # 任意のパス
    csv_files = os.listdir(data_path)
    csv_files.sort()
    train_csv_files = []
    for index in train_index_list:
        train_csv_files.append(csv_files[index])
    input_data=[]
    for file in train_csv_files:
        single_data=pd.read_csv(os.path.join(data_path, file),skiprows=1).values
        input_data.append(decimate(single_data)[:,1:])
    input_array = np.array(input_data)
    mean_list = []
    std_list = []
    for i in range(input_array.shape[2]):
        mean_list.append(input_array[:,:,i].mean())
        std_list.append(input_array[:,:,i].std())
    # print("平均", mean_list)
    # print("標準偏差", std_list)
    return mean_list, std_list

class MazdaDataset(Dataset):
    def __init__(self, data):
        super(MazdaDataset, self).__init__()
        self.data = data
    
    def __len__(self):
        return len(self.data['inp'])
    
    def __getitem__(self, index):
        return {
            'inp': torch.Tensor(self.data['inp'][index]),
            'spec': torch.Tensor(self.data['spec'][index]),
            'gt': torch.Tensor(self.data['gt'][index]),
        }

# データセットを作成する関数
def create_dataset(original_data, index_list, is_train, mean_list=[], std_list=[]):
    data = {}
    input_data_list = []
    spec_data_list = []
    gt_data_list = []
    data_name_list = []
    
    for index in index_list:
        input_data_list.append(original_data['inp'][index])
        spec_data_list.append(original_data['spec'][index])
        gt_data_list.append(original_data['gt'][index])
    input_array = np.array(input_data_list)
    spec_array = np.array(spec_data_list)
    gt_array = np.array(gt_data_list)

    # 入力の標準化処理
    if is_train:
        mean_list, std_list = find_meanstd(index_list)
    else:
        mean_list, std_list = mean_list, std_list
    # print(input_array.shape)
    if is_train:
        print('\n\nTrain Normalization')
    else:
        print('\n\nValidation Normalization')
    print('----------------------------------------------')
    print('[input]')
    # 入力[look_back秒分]のデータの標準化
    for i in tqdm(range(input_array.shape[3])):
        input_array[:, :, :, i] = (input_array[:, :, :, i]-mean_list[i])/std_list[i]
    print('\n[spec]')
    # 入力[制御条件]のデータの標準化
    for i in tqdm(range(spec_array.shape[2])):
        spec_array[:, :, i] = (spec_array[:, :, i]-mean_list[i])/std_list[i]
    print('\n[ground truth]')
    # 出力のデータの標準化
    for i in tqdm(range(gt_array.shape[2])):
        gt_array[:, :, i] = (gt_array[:, :, i]-mean_list[i+9])/std_list[i+9]
    
    data['inp'] = rearrange(input_array, 'a b c d -> (a b) c d')
    data['spec'] = rearrange(spec_array, 'a b c -> (a b) c')
    data['gt'] = rearrange(gt_array, 'a b c -> (a b) c')
    
    # print(data['inp'].shape)
    # print(data['spec'].shape)
    # print(data['gt'].shape)
    print('----------------------------------------------')
    
    return MazdaDataset(data), mean_list, std_list