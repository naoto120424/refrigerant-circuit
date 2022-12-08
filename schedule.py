import subprocess

model_list = ['LSTM', 'BaseTransformer']
look_back_list = range(5, 41, 5)
params_list = []
for model in model_list:
    for look_back in look_back_list:
        params_list.append([model, look_back])

for params in params_list:
    command = [
        'python', 'train.py',
        '--model', str(params[0]),
        '--look_back', str(params[1]),
        '--e_name', 'Mazda Refrigerant Circuit Turtrial 20221207_3',
        '--seed', '42',
    ]
    print('\ncommand: ', command)
    subprocess.run(command)