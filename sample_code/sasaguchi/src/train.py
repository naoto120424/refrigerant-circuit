import argparse
from utils.dataloader import create_dataset, split_dataset, MazdaDataset
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split, KFold, StratifiedKFold
import torch
import torch.utils.data
import os
from tqdm import tqdm
import time
import numpy as np
import pandas as pd
from time import time
# from torch.utils.tensorboard import SummaryWriter
from model.vit import ViT
from utils.utils import model_list, criterion_list
from utils.utils import CFG, seed_everything, make_experiment_id_and_path, find_x_rate_ca, write_evaluation_index_explanation, write_file_final_loss_info, write_file_loss_info, to_sort_and_df, save_fig
from utils.utils import save_scaling_parameter, out_overview_of_experiment

def main():
    parser=argparse.ArgumentParser(description='Train the individual Transformer model')
    parser.add_argument('--ds',type=int,default=2000)
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
    parser.add_argument('--epoch',type=int, default=400)
    parser.add_argument('--bs',type=int,default=1)
    parser.add_argument('--lr',type=float,default=1.e-4)
    parser.add_argument('--debug', type=bool, default=False)
    parser.add_argument('--model', type=str, default="basic")
    parser.add_argument('--criterion', type=str, default="mse")
    parser.add_argument('--s_width', type=int, default=10)
    args=parser.parse_args()

    seed_everything()
    experiment_id, experiment_path = make_experiment_id_and_path()
    print('experiment id:', experiment_id)
    if args.debug:
        epoch_num = 3
    else:
        epoch_num = args.epoch

    # dataset = create_dataset(use_std=args.use_std)
    data = create_dataset(data_size=args.ds, stratified_width=args.s_width)
    data_size = len(data['inp'])
    print(f'データ数:{data_size}')
    # KFoldで分割した評価指標をまとめるリスト
    all_sorted_loss_hr_rate_max_list = []
    all_sorted_loss_integrated_hr_list = []
    all_sorted_loss_x50_ca_list = []
    all_sorted_loss_distance_list = []

    kf = StratifiedKFold(n_splits=CFG.FOLD_NUM, shuffle=True)
    for fold_index, (train_index_list, val_index_list) in enumerate(kf.split(data['inp'], data['max_layer'])):
        # if fold_index != 0:
        #     break
        print(f"fold{fold_index+1}")
        # log_path = os.path.join(experiment_path, f'fold{fold_index+1}', 'log')
        # log=SummaryWriter(log_path)
        
        if args.model in model_list:
            model = model_list[args.model]
        elif args.model == 'vit':
            #データをパッチ分割するサイズを指定z
            patch_split_size = [args.x_split, args.y_split, args.z_split]  
            model = ViT(args.preds, patch_split_size, dim=args.emb_size, depth=args.layers, heads=args.heads, fc_dim=2048, dim_head=args.dim_head, dropout=args.dropout, emb_dropout=args.dropout)
        else:
            print("model名が不適です。")
            return
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f'使用デバイス:{device}')
        model.to(device)

        # モデルのパラメータの初期化
        for name, param in model.named_parameters():
            if 'weight' in name:
                torch.nn.init.normal_(param, mean=0, std=0.01)
                # print(name, param.data)
            if 'bias' in name:
                torch.nn.init.constant_(param, 0)
                # print(name, param.data)

        optimizer = torch.optim.Adam(model.parameters(), lr=CFG.LR)
        if args.criterion in criterion_list:
            criterion = criterion_list[args.criterion]
        else:
            print("criterion名が不適です。")
            return
        if fold_index == 0:
            out_overview_of_experiment(experiment_id, experiment_path, model.name, criterion.name, data_size, epoch_num, args.bs)

        
        # if args.resume_train:
        #     model.load_state_dict(torch.load(f'models/{args.name}/{args.model_pth}'))

        train_dataset, mean_state_quantity, std_state_quantity, mean_spec, std_spec = split_dataset(data, train_index_list, model.is_replaced, is_train=True)
        val_dataset, _, _, _, _ = split_dataset(data, val_index_list, model.is_replaced, is_train=False, mean_state_quantity=mean_state_quantity, std_state_quantity=std_state_quantity, mean_spec=mean_spec, std_spec=std_spec)

        # DataLoaderを作成
        tr_dl = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
        val_dl = DataLoader(val_dataset, batch_size=args.bs, shuffle=False)

        # scaling_parameterの保存
        save_scaling_parameter(os.path.join(experiment_path, f'fold{fold_index+1}', 'scaling_parameter', 'mean_state_quantity.txt'), mean_state_quantity)
        save_scaling_parameter(os.path.join(experiment_path, f'fold{fold_index+1}', 'scaling_parameter', 'std_state_quantity.txt'), std_state_quantity)
        save_scaling_parameter(os.path.join(experiment_path, f'fold{fold_index+1}', 'scaling_parameter', 'mean_spec.txt'), mean_spec)
        save_scaling_parameter(os.path.join(experiment_path, f'fold{fold_index+1}', 'scaling_parameter', 'std_spec.txt'), std_spec)

        best_loss = 100.0
        best_epoch_num = 0
        for epoch in range(epoch_num):
            print('-------------')
            print(f'Epoch {epoch+1}/{epoch_num}(fold{fold_index+1})')
            # トレーニング
            epoch_loss=0
            model.train()
                
            for batch in tqdm(tr_dl, position=0, dynamic_ncols=True):
                optimizer.zero_grad()
                inp = batch['inp'].to(device)
                gt = batch['gt'].to(device)
                spec = batch['spec'].to(device)

                weight = batch['weight'].float().to(device)

                pred = model(inp, spec)

                # 損失関数の種類によってここが変わる
                if criterion.is_weighted:
                    weight = batch['weight'].float().to(device)
                    loss = criterion(pred, gt, weight) # 重み付き
                else:
                    loss = criterion(pred, gt) # L1, MSE, SmoothL1, RMSLE
                #loss = criterion(pred, gt, weight, threshold_num) # 閾値あり重み付き
                
                #print(gt.shape)
                #print(pred.shape)
                loss.backward()
                optimizer.step()
                # print("train epoch %03i/%03i  batch %04i / %04i loss: %7.4f" % (epoch, args.epoch, id_b, len(tr_dl), loss.item()))
                epoch_loss += loss.item() * inp.size(0)
            epoch_loss = epoch_loss / len(tr_dl)
            # log.add_scalar('Loss/train', epoch_loss, epoch)
            print(f'train Loss: {epoch_loss}')

            with torch.no_grad():
                # テスト
                model.eval()

                epoch_test_error=0

                for batch in tqdm(val_dl, position=0, dynamic_ncols=True):
                    inp = batch['inp'].to(device)
                    gt = batch['gt'].to(device)
                    spec = batch['spec'].to(device)
                    
                    pred = model(inp, spec)

                    test_error = torch.mean(torch.abs(gt - pred))
                    # print("test epoch %03i/%03i  batch %04i / %04i" % (
                    #         epoch, args.epoch, id_b, len(val_dl)))
                    epoch_test_error += test_error.item() * inp.size(0)
                
                epoch_test_error = epoch_test_error / len(val_dl)
                # log.add_scalar('Error/test', epoch_test_error, epoch)
                print(f'val Loss: {epoch_test_error}')

                #test_loss_path = pathlib.Path(f'eval/{args.name}/test/error.txt')
                # test_loss_path = pathlib.Path(f'eval/{args.name}/error.txt')
                # with open(test_loss_path, mode='a') as f:
                #     print("%03i %7.4f" % (epoch, epoch_test_error / len(val_dl)) , file=f)

            # if (epoch == 0) or (epoch % 10 == args.save_step):     
            #     torch.save(model.state_dict(),f'models/{args.name}/{epoch:05d}.pth')
            if (epoch_test_error < best_loss):
                    best_epoch_num = epoch
                    best_loss = epoch_test_error
                    print('This is the best model.')
                    print('-----SAVE-----')
                    model_path = os.path.join(experiment_path, f'fold{fold_index+1}', 'weight', 'best_model.pth')
                    torch.save(model.state_dict(), model_path)

        with open(os.path.join(experiment_path, f'fold{fold_index+1}', 'weight', 'best_model_num.txt'), 'w') as f:
            f.write(f'best_epoch_num:{best_epoch_num}')

        model_path = os.path.join(experiment_path, f'fold{fold_index+1}', 'weight', 'best_model.pth')
        loss_hr_rate_max_list = []
        loss_integrated_hr_list = []
        loss_x50_ca_list = []
        loss_distance_list = []
        output_csv_list = []
        output_csv_columns = ['case'] + [i for i in range(101)]
        print(f"fold-{fold_index+1} 検証開始")
        with torch.no_grad():
            model.load_state_dict(torch.load(model_path))
            model.eval()
            for val_index in tqdm(val_index_list, dynamic_ncols=True):
                # case = os.path.basename(state_quantity_path)[:-6]
                case = data['data_name'][val_index]
                case_num = int(case[4:].lstrip('0'))
                save_path = os.path.join(experiment_path, f'fold{fold_index+1}', 'figure', case)
                os.makedirs(save_path, exist_ok=True)
                state_quantity = data['inp'][val_index]
                
                state_quantity = (state_quantity - mean_state_quantity) / std_state_quantity
                state_quantity = state_quantity.astype(np.float32)
                if model.is_replaced:
                    state_quantity = state_quantity.transpose(3, 0, 1, 2)
                state_quantity = torch.from_numpy(state_quantity).clone()
                # unsqueeze(0): (12, 40, 40, 20) or (40, 40, 20, 12) => (1, 12, 40, 40, 20) or (1, 40, 40, 20, 12)
                state_quantity = state_quantity.unsqueeze(0).to(device)

                spec = data['spec'][val_index]
                spec = (spec - mean_spec) / std_spec
                spec = spec.astype(np.float32)
                spec = torch.from_numpy(spec).clone()
                # unsqueeze(0): (7) => (1, 7)
                spec = spec.unsqueeze(0).to(device)

                pred = model(state_quantity, spec).detach().to('cpu').numpy().copy()[0]
                gt = data['gt'][val_index]

                output_csv_list.append([case] + pred.tolist())
                pd_output_csv = pd.DataFrame(output_csv_list, columns=output_csv_columns).set_index('case').T
                pd_output_csv.index.name = 'time'
                pd_output_csv.to_csv(os.path.join(experiment_path, 'csv', f'output_fold{fold_index+1}.csv'))

                # 評価指標1. HR_Rate(max):燃焼の激しさの度合いを評価するもの
                gt_hr_rate_max = gt.max()
                pred_hr_rate_max = pred.max()
                loss_hr_rate_max = abs(gt_hr_rate_max-pred_hr_rate_max)
                loss_hr_rate_max_list.append([case_num, loss_hr_rate_max])

                # 評価指標2. Integrated_HR : 投入熱量に対してどれだけ燃えたかを評価するもの(HR_Rateを全て足したもの)
                gt_integrated_hr = gt.sum()
                pred_integrated_hr = pred.sum()
                loss_integrated_hr = abs(gt_integrated_hr-pred_integrated_hr)
                loss_integrated_hr_list.append([case_num, loss_integrated_hr])

                # 評価指標3. X50(deg) : 燃焼がクランクアングルのなかでどのあたりで主におこなわれたかを評価するもの
                gt_x50_ca = find_x_rate_ca(gt, gt_integrated_hr, 0.5)
                pred_x50_ca = find_x_rate_ca(pred, pred_integrated_hr, 0.5)
                if (gt_x50_ca is  None) or (pred_x50_ca is  None): # 割合取得がエラーになった場合
                    with open(os.path.join(save_path, 'loss.txt'), 'w') as f:
                        f.write(f'fold{fold_index+1}_{case}\n')
                        f.write('---------------------------\n')
                        f.write('gt\n')
                        f.write(f'評価指標1: {gt_hr_rate_max}\n')
                        f.write(f'評価指標2: {gt_integrated_hr}\n')
                        f.write('---------------------------\n')
                        f.write('prediction\n')
                        f.write(f'評価指標1: {pred_hr_rate_max}\n')
                        f.write(f'評価指標2: {pred_integrated_hr}\n')
                        f.write('---------------------------\n')
                        f.write('誤差\n')
                        f.write(f'評価指標1: {loss_hr_rate_max}\n')
                        f.write(f'評価指標2: {loss_integrated_hr}\n')
                        f.write('評価指標3, 4はエラーにより表示できないよ!\n')
                        write_evaluation_index_explanation(f)
                        f.close()
                    continue
                loss_x50_ca = abs(gt_x50_ca - pred_x50_ca)
                loss_x50_ca_list.append([case_num, loss_x50_ca])

                # 評価指標4. X10-X90(deg) : 燃焼開始から終了までに要した期間を評価するもの
                gt_x90_ca = find_x_rate_ca(gt, gt_integrated_hr, 0.9)
                gt_x10_ca = find_x_rate_ca(gt, gt_integrated_hr, 0.1)
                pred_x90_ca = find_x_rate_ca(pred, pred_integrated_hr, 0.9)
                pred_x10_ca = find_x_rate_ca(pred, pred_integrated_hr, 0.1)
                if (gt_x90_ca is None) or (pred_x90_ca is None) or (gt_x10_ca is None) or (pred_x10_ca is None): # 割合取得がエラーになった場合
                    with open(os.path.join(save_path, 'loss.txt'), 'w') as f:
                        f.write(f'fold{fold_index+1}_{case}\n')
                        f.write('---------------------------\n')
                        f.write('gt\n')
                        f.write(f'評価指標1: {gt_hr_rate_max}\n')
                        f.write(f'評価指標2: {gt_integrated_hr}\n')
                        f.write(f'評価指標3: {gt_x50_ca}\n')
                        f.write('---------------------------\n')
                        f.write('prediction\n')
                        f.write(f'評価指標1: {pred_hr_rate_max}\n')
                        f.write(f'評価指標2: {pred_integrated_hr}\n')
                        f.write(f'評価指標3: {pred_x50_ca}\n')
                        f.write('---------------------------\n')
                        f.write('誤差\n')
                        f.write(f'評価指標1: {loss_hr_rate_max}\n')
                        f.write(f'評価指標2: {loss_integrated_hr}\n')
                        f.write(f'評価指標3: {loss_x50_ca}\n')
                        f.write('評価指標4はエラーにより表示できないよ!\n')
                        write_evaluation_index_explanation(f)
                        f.close()
                    continue
                pred_distance = pred_x90_ca - pred_x10_ca
                gt_distance = gt_x90_ca - gt_x10_ca
                loss_distance = abs(gt_distance-pred_distance)
                loss_distance_list.append([case_num, loss_distance])

                with open(os.path.join(save_path, 'loss.txt'), 'w') as f:
                    f.write(f'fold{fold_index+1}_{case}\n')
                    f.write('---------------------------\n')
                    f.write('gt\n')
                    f.write(f'評価指標1: {gt_hr_rate_max}\n')
                    f.write(f'評価指標2: {gt_integrated_hr}\n')
                    f.write(f'評価指標3: {gt_x50_ca}\n')
                    f.write(f'評価指標4: {gt_distance}\n')
                    f.write('---------------------------\n')
                    f.write('prediction\n')
                    f.write(f'評価指標1: {pred_hr_rate_max}\n')
                    f.write(f'評価指標2: {pred_integrated_hr}\n')
                    f.write(f'評価指標3: {pred_x50_ca}\n')
                    f.write(f'評価指標4: {pred_distance}\n')
                    f.write('---------------------------\n')
                    f.write('誤差\n')
                    f.write(f'評価指標1: {loss_hr_rate_max}\n')
                    f.write(f'評価指標2: {loss_integrated_hr}\n')
                    f.write(f'評価指標3: {loss_x50_ca}\n')
                    f.write(f'評価指標4: {loss_distance}\n')
                    write_evaluation_index_explanation(f)
                    f.close()

                save_fig(gt, pred, os.path.join(save_path, f'{case}.png'), title=case, xlabel='Crank Angle(deg)', ylabel='Heat Release Rate(J/deg)', label1='gt', label2='pred')

            # loss_listをソートして、dataframeに変更        
            df_sorted_loss_hr_rate_max_list = to_sort_and_df(loss_hr_rate_max_list, 'hr_rate_max')
            df_sorted_loss_hr_rate_max_list.to_csv(os.path.join(experiment_path, f'fold{fold_index+1}', 'loss', 'loss_1_hr_rate_max.csv'))
            df_sorted_loss_integrated_hr_list = to_sort_and_df(loss_integrated_hr_list, 'integrated_hr')
            df_sorted_loss_integrated_hr_list.to_csv(os.path.join(experiment_path, f'fold{fold_index+1}', 'loss', 'loss_2_integrated_hr.csv'))
            df_sorted_loss_x50_ca_list = to_sort_and_df(loss_x50_ca_list, 'x_50_ca')
            df_sorted_loss_x50_ca_list.to_csv(os.path.join(experiment_path, f'fold{fold_index+1}', 'loss', 'loss_3_x50_ca.csv'))
            df_sorted_loss_distance_list = to_sort_and_df(loss_distance_list, 'distance')
            df_sorted_loss_distance_list.to_csv(os.path.join(experiment_path, f'fold{fold_index+1}', 'loss', 'loss_4_distance.csv'))

            loss_info_path = os.path.join(experiment_path, f'fold{fold_index+1}', f'fold{fold_index+1}_loss_info.txt')
            with open(loss_info_path, 'w') as f:
                f.write(f'評価指標の誤差まとめ(fold{fold_index+1})\n')
                f.write('---------------------------\n')
                f.write('評価指標1. HR_Rate_Max : 燃焼の激しさの度合いを評価するもの\n')
                write_file_loss_info(f, df_sorted_loss_hr_rate_max_list)
                f.write('---------------------------\n')
                f.write('評価指標2. Integrated_HR : 投入熱量に対してどれだけ燃えたかを評価するもの(HR_Rateを全て足したもの)\n')
                write_file_loss_info(f, df_sorted_loss_integrated_hr_list)
                f.write('---------------------------\n')
                f.write('評価指標3. X50(deg) : 燃焼がクランクアングルのなかでどのあたりで主におこなわれたかを評価するもの\n')
                write_file_loss_info(f, df_sorted_loss_x50_ca_list)
                f.write('---------------------------\n')
                f.write('評価指標4. X10-X90(deg) : 燃焼開始から終了までに要した期間を評価するもの\n')
                write_file_loss_info(f, df_sorted_loss_distance_list)
                write_evaluation_index_explanation(f)
                f.close()
            all_sorted_loss_hr_rate_max_list.append(df_sorted_loss_hr_rate_max_list.values.T.tolist()[0])
            all_sorted_loss_integrated_hr_list.append(df_sorted_loss_integrated_hr_list.values.T.tolist()[0])
            all_sorted_loss_x50_ca_list.append(df_sorted_loss_x50_ca_list.values.T.tolist()[0])
            all_sorted_loss_distance_list.append(df_sorted_loss_distance_list.values.T.tolist()[0])

    final_loss_info_path = os.path.join(experiment_path, 'final_loss_info.txt')
    with open(final_loss_info_path, 'w') as f:
        f.write(f'評価指標の誤差まとめ\n')
        f.write('---------------------------\n')
        f.write('評価指標1. HR_Rate_Max : 燃焼の激しさの度合いを評価するもの\n')
        write_file_final_loss_info(f, all_sorted_loss_hr_rate_max_list)
        f.write('---------------------------\n')
        f.write('評価指標2. Integrated_HR : 投入熱量に対してどれだけ燃えたかを評価するもの(HR_Rateを全て足したもの)\n')
        write_file_final_loss_info(f, all_sorted_loss_integrated_hr_list)
        f.write('---------------------------\n')
        f.write('評価指標3. X50(deg) : 燃焼がクランクアングルのなかでどのあたりで主におこなわれたかを評価するもの\n')
        write_file_final_loss_info(f, all_sorted_loss_x50_ca_list)
        f.write('---------------------------\n')
        f.write('評価指標4. X10-X90(deg) : 燃焼開始から終了までに要した期間を評価するもの\n')
        write_file_final_loss_info(f, all_sorted_loss_distance_list)
        write_evaluation_index_explanation(f)
        f.close()


if __name__=='__main__':
    start_time = time()
    main()
    print('-------------')
    print('Elapsed time: {}'.format(time()-start_time))
