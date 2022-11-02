from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
import torch


def create_dataset(use_std=False, use_log=False):
    path = "../Mazda_1400/"
    data_path = os.path.join(path,"data/")
    datasets_list = os.listdir(data_path)
    datasets_list.sort()

    data = {}
    input_data = []
    output_data = []
    weight_list = []
    threshold_num = []
    data_name = []

    print("start loading dataset")

    for i_dt, dt in enumerate(datasets_list):
        # input data
        print("%03i / %03i - loading %s"%(i_dt+1,len(datasets_list),dt))
        inp = np.load(os.path.join(data_path, dt, dt + "_2.npy")) # 40x40x20x12
 
        input_data.append(inp)    
        
        # output data
        out = pd.read_csv(os.path.join(data_path, dt, "thermo1deg.out"), delimiter="\t", names=["deg", "out"], na_values="?")
        
        if(use_log):
            out['out'] = out['out'] - (-7.638524634256875) # 対数を使う場合は全データ中の最小値で引く

        output_data.append(out['out'])

        weight = out['out'] / out['out'].max()

        threshold = 0.05
        threshold_num.append((weight < threshold).sum())

        weight_list.append(weight)


        data_name.append(dt)

    spec = pd.read_csv(os.path.join(path, "engine_spec_data.csv"), delimiter=",", header=0, index_col = "ID", na_values="?")
    spec_array = np.array(spec.values, dtype = 'float64')

    if(use_std):  
        input_array = np.array(input_data)
        for i in range(input_array.shape[4]):
            mean = input_array[:,:,:,:,i].mean()
            std = input_array[:,:,:,:,i].std()
            input_array[:,:,:,:,i] = (input_array[:,:,:,:,i] - mean) / std
        
        for i in range(spec_array.shape[1]):
            mean = spec_array[:,i].mean()
            std = spec_array[:,i].std()
            spec_array[:,i] = (spec_array[:,i] - mean) / std

    data['inp'] = input_array
    data['gt'] = output_data
    data['spec'] = spec_array
    data['weight'] = weight_list
    data['threshold_num'] = threshold_num
    data['data_name'] = data_name

    return MazdaDataset(data)

class MazdaDataset(Dataset):
    def __init__(self,data):
        super(MazdaDataset,self).__init__()

        self.data=data

    def __len__(self):
        return len(self.data['inp'])


    def __getitem__(self,index):
        return {'inp':torch.Tensor(self.data['inp'][index]),
                'gt':torch.Tensor(self.data['gt'][index]),
                'spec':torch.Tensor(self.data['spec'][index]),
                'weight':torch.Tensor(self.data['weight'][index]),
                'threshold_num':torch.Tensor(self.data['threshold_num'][index]),
                'data_name':self.data['data_name'][index],
                }