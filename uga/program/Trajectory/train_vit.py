import argparse
import baselineUtils
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
import os
import pathlib
import shutil
import time
from transformer.batch import subsequent_mask
from torch.optim import Adam,SGD, RMSprop, Adagrad
from transformer.noam_opt import NoamOpt
import numpy as np
import scipy.io
import json
import pickle

from tqdm import tqdm

from torch.utils.tensorboard import SummaryWriter




def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--dataset_folder',type=str,default='datasets')
    parser.add_argument('--dataset_name',type=str,default='eth')
    parser.add_argument('--obs',type=int,default=8)
    parser.add_argument('--preds',type=int,default=12)
    parser.add_argument('--emb_size',type=int,default=512)
    parser.add_argument('--heads',type=int, default=12)
    parser.add_argument('--layers',type=int,default=6)
    parser.add_argument('--dropout',type=float,default=0.1)
    parser.add_argument('--cpu',action='store_true')
    parser.add_argument('--val_size',type=int, default=0)
    parser.add_argument('--verbose',action='store_true')
    parser.add_argument('--max_epoch',type=int, default=1500)
    parser.add_argument('--batch_size',type=int,default=70)
    parser.add_argument('--validation_epoch_start', type=int, default=30)
    parser.add_argument('--resume_train',action='store_true')
    parser.add_argument('--delim',type=str,default='\t')
    parser.add_argument('--name', type=str, default="zara1")
    parser.add_argument('--factor', type=float, default=1.)
    parser.add_argument('--save_step', type=int, default=1)
    parser.add_argument('--warmup', type=int, default=10)
    parser.add_argument('--evaluate', type=bool, default=True)
    parser.add_argument('--model_pth', type=str)




    args=parser.parse_args()
    model_name=args.name

    try:
        os.mkdir('models')
    except:
        pass
    try:
        os.mkdir('output')
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
        os.mkdir(f'output/{args.name}')
    except:
        pass

    try:
        os.mkdir(f'eval/{args.name}')
    except:
        pass
    try:
        os.mkdir(f'eval/{args.name}/val')
    except:
        pass
    try:
        os.mkdir(f'eval/{args.name}/test')
    except:
        pass
    
    log_file = pathlib.Path(f'logs/%s'%model_name)
    if log_file.exists():
        shutil.rmtree('logs/%s'%model_name)
    log=SummaryWriter('logs/%s'%model_name)


    log.add_scalar('eval/mad', 0, 0)
    log.add_scalar('eval/fad', 0, 0)
    device=torch.device("cuda")

    if args.cpu or not torch.cuda.is_available():
        device=torch.device("cpu")

    args.verbose=True


    ## creation of the dataloaders for train and validation
    if args.val_size==0:
        train_dataset,_ = baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs+1,args.preds,delim=args.delim,train=True,verbose=args.verbose)
        val_dataset, _ = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, 0, args.obs+1,
                                                                    args.preds, delim=args.delim, train=False,
                                                                    verbose=args.verbose)
    else:
        train_dataset, val_dataset = baselineUtils.create_dataset(args.dataset_folder, args.dataset_name, args.val_size,args.obs+1,
                                                              args.preds, delim=args.delim, train=True,
                                                              verbose=args.verbose)

    test_dataset,_ =  baselineUtils.create_dataset(args.dataset_folder,args.dataset_name,0,args.obs+1,args.preds,delim=args.delim,train=False,eval=True,verbose=args.verbose)




    import vit
    model=vit.ViT(in_size=2, num_obs=args.obs, out_size=3, dim=args.emb_size, depth=args.layers, heads=args.heads, mlp_dim=2048,
                 dropout=args.dropout,emb_dropout = args.dropout).to(device)
    if args.resume_train:
        model.load_state_dict(torch.load(f'models/{args.name}/{args.model_pth}'))

    tr_dl = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    val_dl = torch.utils.data.DataLoader(val_dataset, batch_size=args.batch_size, shuffle=True, num_workers=0)
    test_dl = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, num_workers=0)

    #optim = SGD(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01)
    #sched=torch.optim.lr_scheduler.StepLR(optim,0.0005)
    optim = NoamOpt(args.emb_size, args.factor, len(tr_dl)*args.warmup,
                        torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9))
    #optim=Adagrad(list(a.parameters())+list(model.parameters())+list(generator.parameters()),lr=0.01,lr_decay=0.001)
    epoch=0


    #mean=train_dataset[:]['src'][:,1:,2:4].mean((0,1))
    mean=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).mean((0,1))
    #std=train_dataset[:]['src'][:,1:,2:4].std((0,1))
    std=torch.cat((train_dataset[:]['src'][:,1:,2:4],train_dataset[:]['trg'][:,:,2:4]),1).std((0,1))
    means=[]
    stds=[]
    for i in np.unique(train_dataset[:]['dataset']):
        ind=train_dataset[:]['dataset']==i
        means.append(torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).mean((0, 1)))
        stds.append(
            torch.cat((train_dataset[:]['src'][ind, 1:, 2:4], train_dataset[:]['trg'][ind, :, 2:4]), 1).std((0, 1)))
    mean=torch.stack(means).mean(0)
    std=torch.stack(stds).mean(0)

    scipy.io.savemat(f'models/{args.name}/norm.mat',{'mean':mean.cpu().numpy(),'std':std.cpu().numpy()})


    while epoch<args.max_epoch:
        epoch_loss=0
        model.train()

        for id_b,batch in enumerate(tr_dl):

            optim.optimizer.zero_grad()
            inp=(batch['src'][:,1:,2:4].to(device)-mean.to(device))/std.to(device)
            target=(batch['trg'][:,:-1,2:4].to(device)-mean.to(device))/std.to(device)
            target_c=torch.zeros((target.shape[0],target.shape[1],1)).to(device)
            target=torch.cat((target,target_c),-1)
            start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(target.shape[0],1,1).to(device)

            # print("input:", inp.shape)

            pred=model(inp)
        
            # print("pred:", pred.shape)

            loss = F.pairwise_distance(pred[:, -12:,0:2].contiguous().view(-1, 2),
                                       ((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(torch.abs(pred[:, -12:,2]))
            loss.backward()
            optim.step()
            print("train epoch %03i/%03i  batch %04i / %04i loss: %7.4f" % (epoch, args.max_epoch, id_b, len(tr_dl), loss.item()))
            epoch_loss += loss.item()
        #sched.step()
        log.add_scalar('Loss/train', epoch_loss / len(tr_dl), epoch)

        with torch.no_grad():
            model.eval()

            epoch_loss_val=0
            step=0
            model.eval()
            gt = []
            pr = []
            inp_ = []
            peds = []
            frames = []
            dt = []

            for id_b, batch in enumerate(val_dl):
                inp_.append(batch['src'])
                gt.append(batch['trg'][:, :, 0:2])
                frames.append(batch['frames'])
                peds.append(batch['peds'])
                dt.append(batch['dataset'])

                inp = (batch['src'][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(device)
                start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                    device)
                preds = start_of_seq
               

                for i in range(args.preds):
                    out = model(inp)
                    # print("val_out:", out.shape)
                    # print("val_preds:", preds.shape)
                    preds = torch.cat((preds, out[:, -1:, :]), 1)
                    

                loss_val = F.pairwise_distance(preds[:, 1:, 0:2].contiguous().view(-1, 2),
                                       ((batch['trg'][:, :, 2:4].to(device)-mean.to(device))/std.to(device)).contiguous().view(-1, 2).to(device)).mean() + torch.mean(torch.abs(preds[:, 1:,2]))
          
                print("val epoch %03i/%03i  batch %04i / %04i loss: %7.4f" % (epoch, args.max_epoch, id_b, len(val_dl), loss_val.item()))
                epoch_loss_val += loss_val.item()

                preds_tr_b = (preds[:, 1:, 0:2] * std.to(device) + mean.to(device)).cpu().numpy().cumsum(1) + batch['src'][:, -1:,0:2].cpu().numpy()
                pr.append(preds_tr_b)
                print("val epoch %03i/%03i  batch %04i / %04i loss: %7.4f" % (
                    epoch, args.max_epoch, id_b, len(val_dl), loss_val.item()))


            peds = np.concatenate(peds, 0)
            frames = np.concatenate(frames, 0)
            dt = np.concatenate(dt, 0)
            gt = np.concatenate(gt, 0)
            dt_names = test_dataset.data['dataset_name']
            pr = np.concatenate(pr, 0)
            mad, fad, errs = baselineUtils.distance_metrics(gt, pr)
            log.add_scalar('validation/mad', mad, epoch)
            log.add_scalar('validation/fad', fad, epoch)

            val_mad = pathlib.Path(f'eval/{args.name}/val/mad.txt')
            with val_mad.open(mode='w') as f:
                f.write("%03i %7.4f" % (epoch, mad))

            val_fad = pathlib.Path(f'eval/{args.name}/val/fad.txt')
            with val_fad.open(mode='w') as f:
                f.write("%03i %7.4f" % (epoch, fad))


            if args.evaluate:

                model.eval()
                gt = []
                pr = []
                inp_ = []
                preds_ = []
                peds = []
                frames = []
                dt = []
                
                for id_b,batch in enumerate(test_dl):
                    inp_.append(batch['src'])
                    gt.append(batch['trg'][:,:,0:2])
                    frames.append(batch['frames'])
                    peds.append(batch['peds'])
                    dt.append(batch['dataset'])

                    inp = (batch['src'][:, 1:, 2:4].to(device) - mean.to(device)) / std.to(device)
                    start_of_seq = torch.Tensor([0, 0, 1]).unsqueeze(0).unsqueeze(1).repeat(inp.shape[0], 1, 1).to(
                        device)
                    preds_=start_of_seq

                    for i in range(args.preds):
                        out = model(inp)
                        # print("test_out:", out.shape)
                        # print("test_preds:", preds_.shape)
                        preds_=torch.cat((preds_, out[:,-1:,:]),1)
                
                    preds_tr_b=(preds_[:,1:,0:2]*std.to(device)+mean.to(device)).cpu().numpy().cumsum(1)+batch['src'][:,-1:,0:2].cpu().numpy()
                    pr.append(preds_tr_b)
                    print("test epoch %03i/%03i  batch %04i / %04i" % (
                    epoch, args.max_epoch, id_b, len(test_dl)))

                peds = np.concatenate(peds, 0)
                frames = np.concatenate(frames, 0)
                dt = np.concatenate(dt, 0)
                gt = np.concatenate(gt, 0)
                dt_names = test_dataset.data['dataset_name']
                pr = np.concatenate(pr, 0)
                mad, fad, errs = baselineUtils.distance_metrics(gt, pr)


                log.add_scalar('test/mad', mad, epoch)
                log.add_scalar('test/fad', fad, epoch)

                test_mad = pathlib.Path(f'eval/{args.name}/test/mad.txt')
                with test_mad.open(mode='w') as f:
                    f.write("%03i %7.4f" % (epoch, mad))

                test_fad = pathlib.Path(f'eval/{args.name}/test/fad.txt')
                with test_fad.open(mode='w') as f:
                    f.write("%03i %7.4f" % (epoch, fad))

                # log.add_scalar('eval/DET_mad', mad, epoch)
                # log.add_scalar('eval/DET_fad', fad, epoch)

                scipy.io.savemat(f"output/{args.name}/det_{epoch}.mat",
                                 {'input': inp, 'gt': gt, 'pr': pr, 'peds': peds, 'frames': frames, 'dt': dt,
                                  'dt_names': dt_names})

        if epoch%args.save_step==0:

            torch.save(model.state_dict(),f'models/{args.name}/{epoch:05d}.pth')



        epoch+=1
    ab=1































if __name__=='__main__':
    main()
