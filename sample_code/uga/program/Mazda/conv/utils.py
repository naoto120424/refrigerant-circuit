import torch
from torch import nn

class RMSELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = nn.MSELoss()
        
    def forward(self, pred, gt):
        return torch.sqrt(self.mse(pred, gt))

class RMSLELoss(nn.Module):
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

class  WRMSLELoss(nn.Module):
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
