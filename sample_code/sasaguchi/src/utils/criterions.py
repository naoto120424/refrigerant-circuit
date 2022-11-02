import torch
from torch import nn

class SmoothL1Loss(nn.Module):
    name='SmoothL1Loss'
    is_weighted=False
    def __init__(self):
        super().__init__()
        self.smooth_l1 = nn.SmoothL1Loss()

    def forward(self, preds, gts):
        return self.smooth_l1(preds, gts)

class MeanSquaredErrorLoss(nn.Module):
    name = 'MSELoss'
    is_weighted=False
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, gts):
        return self.mse(preds, gts)

class WeightedMeanSquaredErrorLoss(nn.Module):
    name = 'WMSELoss'
    is_weighted=True
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()

    def forward(self, preds, gts, weights):
        return (weights * (preds - gts) ** 2).mean()

class RootMeanSquaredLogErrorLoss(nn.Module):
    name = 'RMSLELoss'
    is_weighted=False
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, gt):
        return torch.sqrt(self.mse(torch.log(pred + 1), torch.log(gt + 1)))

class  WRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, weight):
        return torch.sqrt((weight * (pred - gt) ** 2).mean())

class  WeightedRootMeanSquaredLogErrorLoss(nn.Module):
    name='WRMSLELoss'
    is_weighted=True
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, weight):
        return torch.sqrt((weight * (torch.log(pred + 1) - torch.log(gt + 1)) ** 2).mean())

class  TH_WRMSELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, weight, threshold_num):
        return torch.sqrt((weight * (pred - gt) ** 2).sum() / threshold_num)

class  TH_WRMSLELoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, pred, gt, weight, threshold_num):
        return torch.sqrt((weight * (torch.log(pred + 1) - torch.log(gt + 1)) ** 2).sum() / threshold_num)
