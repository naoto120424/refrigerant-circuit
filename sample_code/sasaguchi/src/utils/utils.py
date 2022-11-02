from ctypes import sizeof
import os, csv
import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from statistics import mean, median
from tqdm import tqdm
import math
import pickle

from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from model.model import Model
from model.resnet import ResNet3d, Bottleneck
from model.eco_lite import ECO_Lite
from utils.criterions import RootMeanSquaredLogErrorLoss as RMSLELoss, MeanSquaredErrorLoss as MSELoss, SmoothL1Loss
from utils.criterions import WeightedMeanSquaredErrorLoss as WMSELoss, WeightedRootMeanSquaredLogErrorLoss as WRMSLELoss
# from utils.evaluate_function import MeanAbsoluteErrorLoss as MAELoss, RootMeanSquaredErrorLoss as RMSELoss, MeanAbsolutePercentageErrorLoss as MAPELoss, symmetricMeanAbsolutePercentageErrorLoss as sMAPELoss

model_list = {
    'basic': Model(),
    'resnet': ResNet3d(Bottleneck),
    'eco': ECO_Lite(),
}

criterion_list = {
    'mse': MSELoss(),
    'wmse': WMSELoss(),
    'smoothl1': SmoothL1Loss(),
    'rmsle': RMSLELoss(),
    'wrmsle': WRMSLELoss(),
}

# eval_list = {
#     'mae': MAELoss(),
#     'rmse': RMSELoss(),
#     'mape': MAPELoss(),
#     'smape': sMAPELoss(),
# }

class CFG:
    EPOCH = 200
    LR = 1e-4
    WEIGHT_DECAY = 1e-3
    # DATA_DIR ='D:\\Users\\m141102\\Desktop\\広大AI\\02_comd_AI\\test\\dataset'
    DATA_DIR = '../../dataset'
    FOLD_NUM = 5

def make_experiment_id_and_path():
    experiment_path = '../experiment'
    dir_list = os.listdir(experiment_path)
    prev_experiment_id = max([int(x) for x in dir_list])
    current_experiment_id  = prev_experiment_id + 1
    current_experiment_path = os.path.join(experiment_path, str(current_experiment_id))
    os.makedirs(current_experiment_path, exist_ok=True)
    os.makedirs(os.path.join(current_experiment_path, 'csv'), exist_ok=True)
    for i in range(CFG.FOLD_NUM):
        os.makedirs(os.path.join(current_experiment_path, f'fold{i+1}'), exist_ok=True)
        os.makedirs(os.path.join(current_experiment_path, f'fold{i+1}', 'loss'), exist_ok=True)
        os.makedirs(os.path.join(current_experiment_path, f'fold{i+1}', 'log'), exist_ok=True)
        os.makedirs(os.path.join(current_experiment_path, f'fold{i+1}', 'weight'), exist_ok=True)
        os.makedirs(os.path.join(current_experiment_path, f'fold{i+1}', 'figure'), exist_ok=True)
        os.makedirs(os.path.join(current_experiment_path, f'fold{i+1}', 'scaling_parameter'), exist_ok=True)
    return current_experiment_id, current_experiment_path

def out_overview_of_experiment(experiment_id, experiment_path, model_name, loss_name, data_length, epoch_num, bs_num):
    print(f'モデル名: {model_name}')
    print(f'損失関数: {loss_name}')
    with open(os.path.join(experiment_path, 'overview.txt'), 'w') as f:
        f.write(f'experiment id: {experiment_id}\n')
        f.write(f'モデル名: {model_name}\n')
        f.write(f'損失関数: {loss_name}\n')
        f.write(f'最適化関数: Adam\n')
        f.write(f'スケーリング法: 標準化\n')
        f.write(f'データ数: {data_length}\n')
        f.write(f'エポック数: {epoch_num}\n')
        f.write(f'バッチサイズ数: {bs_num}\n')
        f.write(f'交差検証法: StratifiedKFold\n')
        f.write(f'↓自由記述欄↓')
        f.close()

def save_scaling_parameter(save_path, save_list):
    with open(os.path.join(save_path), 'wb') as f:
        pickle.dump(save_list, f)
    # with open(save_path, 'w'):
    #     for

def load_scaling_parameter(path):
    with open(path, 'rb') as f:
        return pickle.load(f)

def to_sort_and_df(loss_list, evaluation_name):
    df_loss_list = pd.DataFrame(loss_list, columns=['case_id', f'{evaluation_name}_loss']).set_index('case_id')
    df_sorted_loss_list = df_loss_list.sort_values(f'{evaluation_name}_loss')
    return df_sorted_loss_list

def new_to_df(f, list):
    df_list = pd.DataFrame(list, columns=['case_id', 'target', 'output']).set_index('case_id')
    r_2 = r2_score(df_list['target'], df_list['output'])
    rmse = np.sqrt(mean_squared_error(df_list['target'], df_list['output']))
    mae = mean_absolute_error(df_list['target'], df_list['output'])
    f.write(f'R2:{r_2}\nRMSE:{rmse}\nMAE:{mae}\n')
    return [r_2, rmse, mae]

def write_file_loss_info(f, df_list):
    max = df_list.max().iloc[-1]
    min = df_list.min().iloc[-1]
    mean = df_list.mean().iloc[-1]
    median = df_list.median().iloc[-1]
    f.write(f'最大値:{max}\n最小値:{min}\n平均値:{mean}\n中央値:{median}\n')

def write_file_final_loss_info1(f, all_list):
    df_all_list = pd.DataFrame(all_list, columns=['R2', 'RMSE', 'MAE'])
    r_2 = df_all_list['R2'].mean()
    rmse = df_all_list['RMSE'].mean()
    mae = df_all_list['MAE'].mean()
    f.write(f'R2:{r_2}\nRMSE:{rmse}\nMAE:{mae}\n')

def write_file_final_loss_info(f, all_sorted_loss_list):
    max_list=[]
    min_list=[]
    mean_list=[]
    median_list=[]
    for i in range(len(all_sorted_loss_list)):
        max_list.append(max(all_sorted_loss_list[i]))
        min_list.append(min(all_sorted_loss_list[i]))
        mean_list.append(mean(all_sorted_loss_list[i]))
        median_list.append(median(all_sorted_loss_list[i]))
    f.write(f'最大値:{np.mean(max_list)} ± {np.std(max_list)}\n最小値:{np.mean(min_list)} ± {np.std(min_list)}\n平均値:{np.mean(mean_list)} ± {np.std(mean_list)}\n中央値:{np.mean(median_list)} ± {np.std(median_list)}\n')

def write_evaluation_index_explanation(f):
    f.write('\n評価指標\n')
    f.write('1. HR_Rate_Max : 燃焼の激しさの度合いを評価するもの\n')
    f.write('2. Integrated_HR : 投入熱量に対してどれだけ燃えたかを評価するもの(HR_Rateを全て足したもの)\n')
    f.write('3. X50(deg) : 燃焼がクランクアングルのなかでどのあたりで主におこなわれたかを評価するもの\n')
    f.write('4. X10-X90(deg) : 燃焼開始から終了までに要した期間を評価するもの\n')
    f.write('X{割合*100}Integrated_HRに対して燃焼開始からどれだけ燃えたかを示す指標\n')
    f.write('ex) 0.1→X10, 0.5→X50, 0.9→X90\n')

def seed_everything(seed=42):
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True


def save_fig(data1, data2, save_path, title, xlabel=None, ylabel=None, label1=None, label2=None, ymax=None):
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1)

    ax.plot(data1,  color='#e46409', label=label1)
    ax.plot(data2, color='b', label=label2)

    ax.set_title(title)
    ax.set_xlabel(xlabel)

    if ymax is not None:
        plt.ylim([0, ymax]) 
    # ax.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax.set_ylabel(ylabel)

    ax.legend(loc='best')
    # ax.set_title('Plot of sine and cosine')

    plt.savefig(save_path)


# スケーリングするため、訓練データの各特長量の最大・最少を求める
# もっとマシな実装方法ありそう
def find_minmax(data_list):
    print('find_minmax')
    # float64にしないと値が小さすぎてmaxが0になる
    max_array = np.full(12, -100).astype(np.float64) # -100は適当
    min_array = np.full(12, 100).astype(np.float64) # 100は適当
    for path in tqdm(data_list, position=0, dynamic_ncols=True):
        array = np.load(path)
        for i in range(12):
            max = array[:, :, :, i].max()
            min = array[:, :, :, i].min()
            max_array[i] = (max if max > max_array[i] else max_array[i])
            min_array[i] = (min if min < min_array[i] else min_array[i])
            
    for i in range(3): # u, v, wの場合マイナスがあるので違う前処理を行う
        max_array[i] = 50
        min_array[i] = 0

    print('max_array')
    print(max_array)
    print('min_array')
    print(min_array)
    return max_array, min_array

def find_meanstd(data_list):
    print('find_meanstd')
    # float64にしないと値が小さすぎてmaxが0になる
    array = []
    mean_array = np.full(12, 0).astype(np.float64)
    std_array = np.full(12, 0).astype(np.float64)
    for path in tqdm(data_list, position=0, dynamic_ncols=True):
        case_array = np.load(path).tolist()
        array.append(case_array)

    nd_array = np.array(array)
    for i in range(12):
        mean_array[i] = nd_array[:, :, :, :, i].mean()
        std_array[i] = nd_array[:, :, :, :, i].std()

    print('mean_array')
    print(mean_array)
    print('std_array')
    print(std_array)
    return mean_array, std_array

def make_dataset(data_list, labels, spec_data):
    return_list_array = []
    return_spec_array = []
    return_labels = []
    id_list = []
    # print(return_array.shape)
    mean_array, std_array = find_meanstd(data_list)

    for array_path in tqdm(data_list, dynamic_ncols=True):
        case = os.path.basename(array_path)[:-6]
        case_num = int(os.path.basename(array_path)[4:].lstrip('0')[:-6])
        array = np.load(array_path)[:, :, :, :]
        array = (array - mean_array) / std_array    # 標準化
        array = array.astype(np.float32)
        ravel_array = np.ravel(array).tolist()   # 1次元化
        # print('X after', ravel_array.shape)
        # print('y', engine_spec_data[i].shape)
        # cat = np.concatenate([ravel_array, engine_spec_data[i]])
        return_list_array.append(ravel_array)

        spec_array = spec_data.loc[case_num].values
        spec_array = spec_array.astype(np.float32)
        spec_array = spec_array.tolist()
        return_spec_array.append(spec_array)

        label = labels.loc[case].values
        label = np.where(label <= 0, 0, label)
        label = label.tolist()
        return_labels.append(label)

        id_list.append(case)

    return pd.DataFrame(return_list_array, index=id_list), pd.DataFrame(return_labels, index=id_list), pd.DataFrame(return_spec_array, index=id_list), id_list

class Dataset(Dataset):
    def __init__(self, file_list, labels, spec_data, output_scaling, max_array=None, min_array=None, mean_array=None, std_array=None):
        self.file_list = file_list  # ファイルパスのリスト
        self.labels = labels
        self.spec_data = spec_data
        self.output_scaling = output_scaling
        self.max_array = max_array
        self.min_array = min_array
        self.mean_array = mean_array
        self.std_array = std_array

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        array_path = self.file_list[index]
        array = np.load(array_path)[:, :, :, :]
        # .\dataset\case0020_2.npy→case0020_2.npy→0020_2.npy→20_2.npy→20
        case = os.path.basename(array_path)[:-6]
        case_num = int(os.path.basename(array_path)[4:].lstrip('0')[:-6])
        
        # array = np.block([array[:, :, :, :], self.spec_data.iloc[case_num-1].values])
        # array = np.block([array[:, :, :, :], self.spec_data.iloc[case_num-1].values])

        if self.max_array is not None:
            array = (array - self.min_array) / (self.max_array - self.min_array)
        else:
            array = (array -self.mean_array) / self.std_array


        spec_array = self.spec_data.loc[case_num].values
        spec_array = spec_array.astype(np.float32)
        spec_tensor = torch.from_numpy(spec_array).clone()

        array = array.astype(np.float32)
        array = array.transpose(3, 0, 1, 2)
        tensor = torch.from_numpy(array)
        label = self.labels.loc[case].values
        label = np.where(label <= 0, 0, label)
        label = torch.tensor(label).float()
        # label = torch.tensor(self.labels.iloc[index].values / 120).float()

        return tensor, label, spec_tensor

def sturges_rule(n):
    return 1 + round(math.log(n, 2))

def hr_rate_info(target):

    hr_max_list = target.max(axis=1)
    hr_max_list = hr_max_list.drop('case1152')

    max_hr_max_list = hr_max_list.max()
    min_hr_max_list = hr_max_list.min()
    mean_hr_max_list = hr_max_list.mean()
    median_hr_max_list = hr_max_list.median()
    
    return max_hr_max_list, min_hr_max_list, mean_hr_max_list, median_hr_max_list

def find_x_rate_ca(list, integrated_hr, rate):
    standard = integrated_hr * rate
    y = 0
    for ca, x in enumerate(list):
        y += x
        if y >= standard:
            return ca

def make_max_layer_list(target, drop=False):
    width = 20
    target_max_layer_list = []
    drop_target_index_list = []
    drop_target_index_int_list = []
    target['max'] = target.max(axis=1)
    if drop:
        width = 5
        drop_target_index_list = target[target['max']>50].index.values.tolist()
        target = target.drop(drop_target_index_list)
        for drop_target_index in drop_target_index_list:
            drop_target_index_int_list.append(int(os.path.basename(drop_target_index)[4:].lstrip('0'))-1)
    len_class = 11
    for index in target.index.values:
        max_layer = int(target.at[index, 'max']/width)
        if max_layer < len_class:
            target_max_layer_list.append(max_layer)
        else:
            target_max_layer_list.append(len_class-1)
    target = target.drop('max', axis=1)
    return target, target_max_layer_list, drop_target_index_int_list
