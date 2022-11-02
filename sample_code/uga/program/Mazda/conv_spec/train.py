import argparse
from dataloader import create_dataset, MazdaDataset
from torch.utils.data import DataLoader
from utils import RMSELoss, RMSLELoss, WRMSELoss, WRMSLELoss, TH_WRMSELoss, TH_WRMSLELoss
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import pathlib
import shutil
import time
from torch.optim import Adam,SGD, RMSprop, Adagrad
import numpy as np
import scipy.io
import json
import pickle

from torch.utils.tensorboard import SummaryWriter

def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--preds',type=int,default=101)
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--dim_head',type=int, default=500)
    parser.add_argument('--layers',type=int, default=3)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--max_epoch',type=int, default=1000)
    parser.add_argument('--batch_size',type=int,default=1)
    parser.add_argument('--lr',type=float,default=1.e-4)
    parser.add_argument('--resume_train',action='store_true')
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="mazda")
    parser.add_argument('--save_step', type=int, default=9)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--use_std', type=bool, default=False)
    parser.add_argument('--use_log', type=bool, default=False)
    parser.add_argument('--model_pth', type=str)
    args=parser.parse_args()
    model_name=args.name


    try:
        os.mkdir('models')
    except:
        pass
    try:
        os.mkdir('eval')
    except:
        pass
    try:
        os.mkdir(f'models/{args.name}')
    except:
        pass
    try:
        os.mkdir(f'eval/{args.name}')
    except:
        pass
    """     
    try:
        os.mkdir(f'eval/{args.name}/val')
    except:
        pass 
    """
    """     
    try:
        os.mkdir(f'eval/{args.name}/test')
    except:
        pass
     """
    log_path = pathlib.Path(f'logs/%s'%model_name)
    if log_path.exists():
        shutil.rmtree(log_path)
    log=SummaryWriter(log_path)

    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    dataset = create_dataset(use_std=args.use_std, use_log=args.use_log)

    # データセットをtrainとvalidationとtestに分割
    """     
    train_size = int( len(dataset) * 0.6 )
    val_size = int( len(dataset) * 0.2 )
    test_size = int( len(dataset) * 0.2 )

    train_data, val_data, test_data = torch.utils.data.random_split(
        dataset,
        [train_size, val_size, test_size],
        generator=torch.Generator().manual_seed(0)  # 乱数シードの固定
    ) 
    """

    # データセットをtrainとtestに分割
    train_size = int( len(dataset) * 0.8 )
    test_size = int( len(dataset) * 0.2 )

    train_data, test_data = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(0)  # 乱数シードの固定
    )

    import vit
    model=vit.ViT(num_pred=args.preds, dim=args.emb_size, depth=args.layers, heads=args.heads, fc_dim=2048, dim_head=args.dim_head, dropout=args.dropout, emb_dropout=args.dropout).to(device)

    if args.resume_train:
        model.load_state_dict(torch.load(f'models/{args.name}/{args.model_pth}'))

    # DataLoaderを作成
    tr_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    #val_dl = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # 損失関数

    #criterion = nn.L1Loss() # L1
    #criterion = nn.MSELoss() # MSE
    #criterion = WRMSELoss() # 重み付き平均二乗誤差
    #criterion = TH_WRMSELoss() # 閾値あり重み付き平均二乗誤差
    #criterion = TH_WRMSLELoss() # 閾値あり重み付き対数平均二乗誤差

    #criterion = nn.SmoothL1Loss() # SmoothL1
    criterion = RMSLELoss() # 対数平均二乗誤差
    #criterion = WRMSLELoss() # 重み付き対数平均二乗誤差
    #criterion = nn.MSELoss() # MSE
    #criterion = RMSELoss() # RMSE

    epoch=0

    while epoch<args.max_epoch:
        # トレーニング
        epoch_loss=0
        model.train()
            
        for id_b,batch in enumerate(tr_dl):
            optimizer.zero_grad()
            inp = batch['inp'].to(device)
            gt = batch['gt'].to(device)
            spec = batch['spec'].to(device)

            weight = batch['weight'].float().to(device)
            threshold_num = batch['threshold_num'].float().to(device)

            pred = model(inp, spec)

            # 損失関数の種類によってここが変わる

            loss = criterion(pred, gt) # L1, MSE, SmoothL1, RMSLE
            #loss = criterion(pred, gt, weight) # 重み付き
            #loss = criterion(pred, gt, weight, threshold_num) # 閾値あり重み付き
            
            #print(gt.shape)
            #print(pred.shape)
            loss.backward()
            optimizer.step()
            print("train epoch %03i/%03i  batch %04i / %04i loss: %7.4f" % (epoch, args.max_epoch, id_b, len(tr_dl), loss.item()))
            epoch_loss += loss.item()
        log.add_scalar('Loss/train', epoch_loss / len(tr_dl), epoch)

        with torch.no_grad():
            
            """ 
            # 検証
            model.eval()

            epoch_loss_val=0

            for id_b, batch in enumerate(val_dl):
                inp = batch['inp'].to(device)
                gt = batch['gt'].to(device)

                pred = model(inp)

                loss_val = torch.mean(torch.abs(gt - pred))
                print("val epoch %03i/%03i  batch %04i / %04i loss: %7.4f" % (
                            epoch, args.max_epoch, id_b, len(val_dl), loss_val.item()))
                epoch_loss_val += loss_val.item()

            log.add_scalar('Loss/validation', epoch_loss_val / len(val_dl), epoch)

            val_loss_path = pathlib.Path(f'eval/{args.name}/val/loss.txt')
            with open(val_loss_path, mode='a') as f:
                print("%03i %7.4f \n" % (epoch, epoch_loss_val/ len(val_dl)), file=f)
            """

            # テスト
            if args.evaluate:
                model.eval()

                epoch_test_error=0

                for id_b,batch in enumerate(test_dl):
                    inp = batch['inp'].to(device)
                    gt = batch['gt'].to(device)
                    spec = batch['spec'].to(device)
                    
                    pred = model(inp, spec)

                    test_error = torch.mean(torch.abs(gt - pred))
                    print("test epoch %03i/%03i  batch %04i / %04i" % (
                            epoch, args.max_epoch, id_b, len(test_dl)))
                    epoch_test_error += test_error.item()

                log.add_scalar('Error/test', epoch_test_error / len(test_dl), epoch)

                #test_loss_path = pathlib.Path(f'eval/{args.name}/test/error.txt')
                test_loss_path = pathlib.Path(f'eval/{args.name}/error.txt')
                with open(test_loss_path, mode='a') as f:
                    print("%03i %7.4f \n" % (epoch, epoch_test_error / len(test_dl)) , file=f)

        if (epoch == 0) or (epoch % 10 == args.save_step):     
            torch.save(model.state_dict(),f'models/{args.name}/{epoch:05d}.pth')

        epoch+=1

if __name__=='__main__':
    main()
