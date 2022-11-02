from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm


def create_dataset(data_size=2000, stratified_width=10):
    path = "../../dataset"
    data_path = os.path.join(path,"input1/")
    datasets_list = os.listdir(data_path)
    datasets_list.sort()
    datasets_list = datasets_list[:data_size]

    data = {}
    input_data = []
    output_data = []
    weight_list = []
    # threshold_num = []
    max_layer_list = []
    data_name = []

    print("データセット読み込み開始")
    for dt in tqdm(datasets_list):
        # input data
        inp = np.load(os.path.join(data_path, dt, dt + "_2.npy")) # 40x40x20x12
 
        input_data.append(inp)    
        
        # output data
        out = pd.read_csv(os.path.join(data_path, dt, "thermo1deg.out"), delimiter="\t", names=["deg", "out"], na_values="?")
        out['out'] = out['out'].where(out['out'] >= 0, 0)
        output_data.append(out['out'])

        weight = out['out'] / out['out'].max()
        weight_list.append(weight)

        # threshold = 0.05
        # threshold_num.append((weight < threshold).sum())

        width = stratified_width
        len_class = 11
        max_layer = int(out['out'].max() / width)
        if max_layer < len_class:
            max_layer_list.append(max_layer)
        else:
            max_layer_list.append(len_class-1)

        data_name.append(dt)
    print("データセット読み込み終了")

    input_array = np.array(input_data)

    spec = pd.read_csv(os.path.join(path, "engine_spec_data.csv"), delimiter=",", header=0, index_col = "ID", na_values="?")
    spec = spec.iloc[:data_size]
    spec_array = np.array(spec.values, dtype = 'float64')

    data['inp'] = input_array
    data['gt'] = output_data
    data['spec'] = spec_array
    data['weight'] = weight_list
    # data['threshold_num'] = threshold_num
    data['max_layer'] = max_layer_list
    data['data_name'] = data_name

    return data

def split_dataset(data, index_list, is_replaced, is_train, mean_state_quantity=[], std_state_quantity=[], mean_spec=[], std_spec=[]):
    dataset = {}
    input_data = []
    spec_data = []
    output_data = []
    weight_list = []
    threshold_num = []
    max_layer_list = []
    data_name = []

    print(f"{'train' if is_train else 'val'} datasetを作成開始")
    if is_train:
        mean_state_quantity = []
        std_state_quantity = []
        mean_spec = []
        std_spec = []
    else:
        mean_state_quantity = mean_state_quantity
        std_state_quantity = std_state_quantity
        mean_spec = mean_spec
        std_spec = std_spec

    for index in index_list:
        input_data.append(data['inp'][index])
        spec_data.append(data['spec'][index])
        output_data.append(data['gt'][index])
        weight_list.append(data['weight'][index])
        # threshold_num.append(data['threshold_num'][index])
        max_layer_list.append(data['max_layer'][index])
        data_name.append(data['data_name'][index])

    input_array = np.array(input_data)
    spec_array = np.array(spec_data)
    # 入力1の標準化処理
    print("入力1の標準化処理開始")
    for i in tqdm(range(input_array.shape[4])):
        if is_train:
            mean = input_array[:,:,:,:,i].mean()
            mean_state_quantity.append(mean)
            std = input_array[:,:,:,:,i].std()
            std_state_quantity.append(std)
        else:
            mean = mean_state_quantity[i]
            std = std_state_quantity[i]

        input_array[:,:,:,:,i] = (input_array[:,:,:,:,i] - mean) / std
    print("入力1の標準化処理終了")
    # 置換処理
    if is_replaced:
        input_array = input_array.transpose(0, 4, 1, 2, 3)    # (1600, 40, 40, 20, 12) => (1600, 12, 40, 40, 20)
    # 入力2の標準化処理
    print("入力2の標準化処理開始")
    for i in tqdm(range(spec_array.shape[1])):
        if is_train:
            mean = spec_array[:,i].mean()
            mean_spec.append(mean)
            std = spec_array[:,i].std()
            std_spec.append(std)
        else:
            mean = mean_spec[i]
            std = std_spec[i]
            
        spec_array[:,i] = (spec_array[:,i] - mean) / std  
    print("入力2の標準化処理終了")

    dataset['inp'] = input_array
    dataset['spec'] = spec_array
    dataset['gt'] = output_data
    dataset['weight'] = weight_list
    # dataset['threshold_num'] = threshold_num
    dataset['max_layer'] = max_layer_list
    dataset['data_name'] = data_name
    
    return MazdaDataset(dataset), mean_state_quantity, std_state_quantity, mean_spec, std_spec

class MazdaDataset(Dataset):
    def __init__(self,data):
        super(MazdaDataset,self).__init__()

        self.data=data

    def __len__(self):
        return len(self.data['inp'])


    def __getitem__(self,index):
        return {
            'inp':torch.Tensor(self.data['inp'][index]),
            'gt':torch.Tensor(self.data['gt'][index]),
            'spec':torch.Tensor(self.data['spec'][index]),
            'weight':torch.Tensor(self.data['weight'][index]),
            # 'threshold_num':torch.Tensor(self.data['threshold_num'][index]),
            'max_layer':self.data['max_layer'][index],
            'data_name':self.data['data_name'][index],
        }