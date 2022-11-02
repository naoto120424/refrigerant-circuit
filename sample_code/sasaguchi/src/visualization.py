import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import glob, os

def main():
    parser=argparse.ArgumentParser(description='visualization')
    parser.add_argument('--e_id',type=int, nargs='*')
    args=parser.parse_args()
    experiment_id_list = args.e_id

    for experiment_id in experiment_id_list:
        print('experiment_id: ', experiment_id)

        # カラムリスト
        path = "../../dataset/mazda/"
        spec = pd.read_csv(os.path.join(path, "engine_spec_data.csv"), delimiter=",", header=0, index_col = "ID", na_values="?")
        spec_columns_list = spec.columns.values.tolist()
        spec_columns_list.append('plane')
        spec_columns_list.append('max')

        gt = pd.read_csv(os.path.join(path, "matome_data.txt"), delim_whitespace=True)
        max_list = gt.max()

        index_list = ['hrrm', 'ihr', 'x50', 'x10_x90']
        soukan_list = {
            'hrrm': [],
            'ihr': [],
            'x50': [],
            'x10_x90': []
        }
        for fold_index in range(5):
            print(f'fold{fold_index+1}')
            dir_path = f'../experiment/{experiment_id}/fold{fold_index+1}'
            dirs=sorted(glob.glob(f'{dir_path}/figure/case*'))
            file_name='/loss.txt'
            num_data = len(dirs)
            columns = ['target', 'prediction', 'loss']
            index = []
            index_num = []
            hrrm = []
            ihr = []
            x50 = []
            x10_x90 = []
            for i in range(num_data):
                case = dirs[i][-8:]
                data_path = dirs[i]+file_name
                index.append(case)
                index_num.append(int(case[4:].lstrip('0')))
                txt_contents = []
                with open(data_path, 'r') as f:
                    for line in f:
                        txt_contents.append(line)
                    hrrm.append([float(txt_contents[3].split()[1]), float(txt_contents[9].split()[1]), float(txt_contents[15].split()[1])])
                    ihr.append([float(txt_contents[4].split()[1]), float(txt_contents[10].split()[1]), float(txt_contents[16].split()[1])])
                    x50.append([float(txt_contents[5].split()[1]), float(txt_contents[11].split()[1]), float(txt_contents[17].split()[1])])
                    x10_x90.append([float(txt_contents[6].split()[1]), float(txt_contents[12].split()[1]), float(txt_contents[18].split()[1])])
            hrrm_df = pd.DataFrame(data=hrrm, index=index, columns=columns)
            ihr_df = pd.DataFrame(data=ihr, index=index, columns=columns)
            x50_df = pd.DataFrame(data=x50, index=index, columns=columns)
            x10_x90_df = pd.DataFrame(data=x10_x90, index=index, columns=columns)
            # print("hrrm", hrrm_df)
            # print("ihr", ihr_df)
            # print("x50", x50_df)
            # print("x10-x90", x10_x90_df)
            # return
            df_list = {
                'hrrm': hrrm_df,
                'ihr': ihr_df,
                'x50': x50_df,
                'x10_x90': x10_x90_df
            }
            for index_name in index_list:
                target_df = df_list[index_name]
                target_df.index.name = 'ID'
                target_df = target_df.reset_index()
                plot_path = os.path.join(dir_path, 'plot', index_name)
                os.makedirs(plot_path, exist_ok=True)
                for spec_column in spec_columns_list:
                    plot_spec_path = os.path.join(plot_path, spec_column)
                    os.makedirs(plot_spec_path, exist_ok=True)

                    # 全体のplot図を作成
                    max = target_df['target'].max() if target_df['target'].max() > target_df['prediction'].max() else target_df['prediction'].max()
                    if spec_column != 'plane':
                        if spec_column == 'max':
                            target_df[spec_column] = max_list.loc[index].values
                        else:
                            target_df[spec_column] = spec[spec_column].loc[index_num].values
                        # sc = ax.scatter(target_df['target'], target_df['prediction'], c=target_df[spec_column], cmap='plasma')
                        # plt.colorbar(sc)
                        fig = go.Figure(
                            data=[
                                go.Scatter(
                                    x=target_df['target'],
                                    y=target_df['prediction'],
                                    mode='markers',
                                    customdata=target_df,
                                    marker=dict(
                                        color=target_df[spec_column],
                                        colorscale='Plasma',  # カラースケール変更
                                        showscale=True  # カラーバーの表示
                                    ),
                                    hovertemplate=(
                                        "%{customdata[0]}<br>"\
                                        "target = %{x}<br>"\
                                        "prediction = %{y}"
                                    ),    
                                ),
                                go.Scatter(
                                    x=[-max//10, max+max//10],
                                    y=[-max//10, max+max//10],
                                    mode='lines',
                                    line=dict(color='gray')
                                )
                            ],
                            layout = go.Layout(
                                title=dict(text= f'{index_name.upper()}_{spec_column}', x=0.5, y=0.9, xanchor='center'),
                                xaxis=dict(title="taregt", range=(-max//10, max+max//10)),
                                yaxis=dict(title="prediction", range=(-max//10, max+max//10), scaleanchor = "x"),
                                height=800,
                                width=800,
                            )
                        )
                        fig.write_image(os.path.join(plot_spec_path, '0_all_plot.png'), engine="kaleido") # test.pngとして保存
                        fig.write_html(os.path.join(plot_spec_path, '0_all_plot.html'))
                    else:
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        ax.axline([0, 0], [1, 1])
                        ax.set_aspect('equal', adjustable='box')
                        ax.set_title(f'{index_name.upper()}_{spec_column.upper()}')
                        ax.set_xlabel('target')
                        ax.set_ylabel('prediction')
                        ax.set_xlim(-1, max+max//10)
                        ax.set_ylim(-1, max+max//10)
                        ax.scatter(target_df['target'], target_df['prediction'], c='#1f77b4')
                        fig.savefig(os.path.join(plot_spec_path, '0_all_plot.png'))
                        plt.close(fig)

                    # 一定範囲ごとにプロットする(必要ないかも)
                    plot_num = 40
                    for i in range(int(len(target_df)/plot_num)):
                        df = target_df.sort_values('target').iloc[i*plot_num : (i+1)*plot_num]
                        if df.empty:
                            # print(f'{i*width} <= target < {(i+1)*width}は空です。')
                            continue
                        max = df['target'].max() if df['target'].max() > df['prediction'].max() else df['prediction'].max()
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        ax.set_aspect('equal', adjustable='box')
                        ax.axline([0, 0], [1, 1])
                        ax.scatter(df['target'], df['prediction'])
                        ax.set_title(f'{index_name.upper()}_{spec_column.upper()} {i+1}/{int(len(target_df)/plot_num)}')
                        ax.set_xlabel('target')
                        ax.set_ylabel('prediction')
                        ax.set_xlim(-1, max+max//10)
                        ax.set_ylim(-1, max+max//10)
                        if spec_column != 'plane':
                            sc = ax.scatter(df['target'], df['prediction'], c=df[spec_column], cmap='plasma')
                            plt.colorbar(sc)
                        else:
                            ax.scatter(df['target'], df['prediction'], c='#1f77b4')
                        fig.savefig(os.path.join(plot_spec_path, f'{i+1}-{int(len(target_df)/plot_num)}.png'))
                        plt.close(fig)
                    
                    if spec_column == 'plane':
                        # 全体の箱ひげ図を作成
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        ax.boxplot(target_df['loss'], vert=False)
                        ax.set_title(f'{index_name.upper()} boxplot')
                        ax.set_xlabel('loss')
                        ax.set_yticklabels('')
                        fig.savefig(os.path.join(plot_spec_path, 'boxplot.png'))
                        plt.close(fig)
                        # 全体の箱ひげ図を作成(外れ値を除く)
                        fig = plt.figure()
                        ax = fig.add_subplot(1, 1, 1)
                        ax.boxplot(target_df['loss'], vert=False, sym="")
                        ax.set_title(f'{index_name.upper()} boxplot')
                        ax.set_xlabel('loss')
                        ax.set_yticklabels('')
                        fig.savefig(os.path.join(plot_spec_path, 'boxplot_without_outliers.png'))
                        plt.close(fig)
                # 相関係数の保存
                soukan = target_df.corr().iat[0, 1]
                soukan_list[index_name].append(soukan)
                with open(os.path.join(plot_path, 'soukan.txt'), 'w') as f:
                    f.write(f'相関係数: {soukan}')
        
        with open(f'../experiment/{experiment_id}/soukan.txt', 'w') as f:
            f.write('相関係数;\n')
            for index_name in index_list:
                f.write(f'{index_name.upper()}: {np.mean(soukan_list[index_name])} ± {np.std(soukan_list[index_name])}\n')

if __name__=='__main__':
    main()