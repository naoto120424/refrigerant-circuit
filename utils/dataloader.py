import torch
from torch.utils.data import Dataset
import numpy as np
import pandas as pd
import os

from tqdm import tqdm
from einops import rearrange


# データ間引き関数
def decimate(data):
    dt = 1
    new_data = []
    pick_time = dt
    for i in range(len(data)):
        if data[i, 0] > pick_time:
            x = data[i, 0], data[i - 1, 0]
            y = data[i, :], data[i - 1, :]
            a, b = np.polyfit(x, y, 1)  # n秒とn+0.1秒間の線形の近似式の傾きa、切片bを求める
            new_data.append([a * pick_time + b])  # n秒のデータを近似式から求める
            pick_time += dt
    new_data = np.array(new_data)
    new_data = np.reshape(new_data, (new_data.shape[0], new_data.shape[2]))
    return new_data


# 全データをロードする関数
def load_data(look_back=20, debug=False):
    print("Load Data")
    print("----------------------------------------------")
    data_path = "../dataset/" if not debug else "../step1_Eva_dataset/"
    csv_files = os.listdir(data_path)
    csv_files.sort()

    num_control_features = 6

    data = {}
    Xdata = []
    Specdata = []
    Ydata = []

    for file in tqdm(csv_files):
        csv_data = pd.read_csv(os.path.join(data_path, file), skiprows=1).values
        single_data = decimate(csv_data)[:, 1:]
        # single_data = csv_data[:, 1:] # No decimate function
        single_data = np.delete(single_data, [8, 6, 5], axis=1)  # Chiller
        spec_data = single_data[:, :num_control_features]
        output_data = single_data[:, num_control_features:]
        input_time_list = []
        spec_list = []
        gt_list = []
        for t in range(single_data.shape[0] - look_back):  # data.shape[0]=1199
            input_time_list.append(single_data[t : t + look_back])
            spec_list.append(spec_data[t + look_back])
            gt_list.append(output_data[t + look_back])
        Xdata.append(np.array(input_time_list))
        Specdata.append(np.array(spec_list))
        Ydata.append(np.array(gt_list))
    data["inp"] = np.array(Xdata)  # shape(case_num, 1199-look_back, look_back, num_all_features)
    data["spec"] = np.array(Specdata)  # shape(case_num, 1199-look_back, num_control_features)
    data["gt"] = np.array(Ydata)  # shape(case_num, 1199-look_back, num_pred_features)
    # print(data["inp"].shape)
    # print(data["spec"].shape)
    # print(data["gt"].shape)

    data["feature_name"] = list(map(lambda x: x.split(".")[0], pd.read_csv(os.path.join(data_path, csv_files[0]), skiprows=0).columns.values))
    data["feature_unit"] = list(map(lambda x: x.split(".")[0], pd.read_csv(os.path.join(data_path, csv_files[0]), skiprows=1).columns.values))
    for i in [9, 7, 6]:  # Chiller
        del data["feature_name"][i]  # Chiller
        del data["feature_unit"][i]  # Chiller
    print("----------------------------------------------")
    return data


# 訓練データから標準化用の平均と標準偏差を求める関数
def find_meanstd(train_index_list, debug=False):
    data_path = "../dataset/" if not debug else "../step1_Eva_dataset/"
    csv_files = os.listdir(data_path)
    csv_files.sort()
    train_csv_files = []
    for index in train_index_list:
        train_csv_files.append(csv_files[index])
    input_data = []
    for file in train_csv_files:
        single_data = pd.read_csv(os.path.join(data_path, file), skiprows=1).values
        input_data.append(decimate(single_data)[:, 1:])
    input_array = np.array(input_data)
    input_array = np.delete(input_array, [8, 6, 5], axis=2)  # Chiller
    mean_list = []
    std_list = []
    for i in range(input_array.shape[2]):
        mean_list.append(input_array[:, :, i].mean())
        std_list.append(input_array[:, :, i].std())
    # print("平均", len(mean_list), mean_list)
    # print("標準偏差", len(std_list), std_list)
    return mean_list, std_list


class MazdaDataset(Dataset):
    def __init__(self, data):
        super(MazdaDataset, self).__init__()
        self.data = data

    def __len__(self):
        return len(self.data["inp"])

    def __getitem__(self, index):
        return {
            "inp": torch.Tensor(self.data["inp"][index]),
            "spec": torch.Tensor(self.data["spec"][index]),
            "gt": torch.Tensor(self.data["gt"][index]),
        }


# データセットを作成する関数
def create_dataset(original_data, index_list, is_train, mean_list=[], std_list=[], debug=False):
    num_control_features = 6  # Chiller
    data = {}
    input_data_list = []
    spec_data_list = []
    gt_data_list = []
    data_name_list = []

    for index in index_list:
        input_data_list.append(original_data["inp"][index])
        spec_data_list.append(original_data["spec"][index])
        gt_data_list.append(original_data["gt"][index])
    input_array = np.array(input_data_list)
    spec_array = np.array(spec_data_list)
    gt_array = np.array(gt_data_list)

    """入力の標準化処理"""
    mean_list, std_list = find_meanstd(index_list, debug) if is_train else (mean_list, std_list)
    print("\n\nTrain Dataset Normalization") if is_train else print("\n\nValidation Dataset Normalization")
    print("----------------------------------------------")

    """入力[look_back秒分]のデータの標準化"""
    print("[input]")
    for i in tqdm(range(input_array.shape[3])):
        input_array[:, :, :, i] = (input_array[:, :, :, i] - mean_list[i]) / std_list[i]

    """入力[制御条件]のデータの標準化"""
    print("\n[spec]")
    for i in tqdm(range(spec_array.shape[2])):
        spec_array[:, :, i] = (spec_array[:, :, i] - mean_list[i]) / std_list[i]

    """出力のデータの標準化"""
    print("\n[ground truth]")
    for i in tqdm(range(gt_array.shape[2])):
        gt_array[:, :, i] = (gt_array[:, :, i] - mean_list[i + num_control_features]) / std_list[i + num_control_features]

    data["inp"] = rearrange(input_array, "a b c d -> (a b) c d")
    data["spec"] = rearrange(spec_array, "a b c -> (a b) c")
    data["gt"] = rearrange(gt_array, "a b c -> (a b) c")

    print("----------------------------------------------")

    return MazdaDataset(data), mean_list, std_list
