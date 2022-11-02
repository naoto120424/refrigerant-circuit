import argparse
from dataloader import create_dataset, MazdaDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import pathlib
from torch.optim import Adam,SGD, RMSprop, Adagrad
import numpy as np
import scipy.io
import matplotlib.pyplot as plt

def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--preds',type=int,default=101)
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=8)
    parser.add_argument('--x_split',type=int, default=1)
    parser.add_argument('--y_split',type=int, default=1)
    parser.add_argument('--z_split',type=int, default=20)
    parser.add_argument('--dim_head',type=int, default=20)
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
        os.mkdir('vis')
    except:
        pass
    try:
        os.mkdir(f'vis/{args.name}')
    except:
        pass
    try:
        os.mkdir(f'vis/{args.name}/{args.model_pth}')
    except:
        pass

    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True

    dataset = create_dataset(use_std=args.use_std, use_log=args.use_log)

    # データセットをtrainとtestに分割
    train_size = int( len(dataset) * 0.8 )
    test_size = int( len(dataset) * 0.2 )

    train_data, test_data = torch.utils.data.random_split(
        dataset,
        [train_size, test_size],
        generator=torch.Generator().manual_seed(0)  # 乱数シードの固定
    )

    #データをパッチ分割するサイズを指定
    patch_split_size = [args.x_split, args.y_split, args.z_split]

    import vit
    model=vit.ViT(num_pred=args.preds, patch_split=patch_split_size, dim=args.emb_size, depth=args.layers, heads=args.heads, fc_dim=2048, dim_head=args.dim_head, dropout=args.dropout, emb_dropout=args.dropout).to(device)

    model.load_state_dict(torch.load(f'models/{args.name}/{args.model_pth}'))

    # DataLoaderを作成
    #tr_dl = DataLoader(train_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    #val_dl = DataLoader(val_data, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = DataLoader(test_data, batch_size=args.batch_size, shuffle=False, num_workers=0)

    with torch.no_grad():

        model.eval()

        test_error=0

        vis_path = pathlib.Path(f'vis/{args.name}/{args.model_pth}')

        x = np.arange(320, 421, 1)    

        for id_b,batch in enumerate(test_dl):
            inp = batch['inp'].to(device)
            gt = batch['gt'].to(device)
            
            pred = model(inp)

            if(args.use_log): # 対数を使う場合は全データ中の最小値を加える
                gt = gt + (-7.638524634256875)
                pred = pred + (-7.638524634256875) 

            error = torch.mean(torch.abs(gt - pred))
            print("batch %04i / %04i" % (id_b, len(test_dl)))
            test_error += error.item()

            
            with open(os.path.join(vis_path, "error.txt"), mode='a') as f:
                print(f"{batch['data_name'][0]} %7.4f \n" % error.item() , file=f)
    

    with open(os.path.join(vis_path, "error.txt"), mode='a') as f:
            print("average error: %7.4f \n" % (test_error / len(test_dl)) , file=f)

if __name__=='__main__':
    main()
