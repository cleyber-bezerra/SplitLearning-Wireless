from matplotlib import pyplot as plt
import pandas as pd
import os
from datetime import datetime

# Inicialize as listas antes de usá-las
train_loss, train_acc, val_loss, val_acc, p_time = [], [], [], [], []
cal_times = []

# Caminho para o arquivo CSV
file = './csv/ia/result_train_sync.csv'

# Carregar o CSV usando Pandas
df = pd.read_csv(file)

# Número de usuários (linhas no DataFrame)
USER = len(df)

# Verificar se o número de linhas no CSV é suficiente
if USER == 0:
    raise ValueError("O arquivo CSV não tem linhas suficientes para o valor de USER fornecido.")

# Popule as listas com os dados do DataFrame
train_loss = df.iloc[:, 1].tolist()
train_acc = df.iloc[:, 2].tolist()
val_loss = df.iloc[:, 3].tolist()
val_acc = df.iloc[:, 4].tolist()
p_time = df.iloc[:, 5].tolist()

for i in range(min(USER, len(val_acc))):
    try:
        val_acc[i] = eval(val_acc[i])
        val_acc[i].insert(0, 0.0)
    except (IndexError, SyntaxError, NameError) as e:
        print(f"Erro ao processar val_acc na posição {i+1}: {e}")
        break

epoch_over_eighty = []
time_over_eighty = []
for user in p_time:
    tmp = 0.0
    user = eval(user)
    cal_time = []
    cal_time.append(0.0)
    for time in user:
        tmp += float(time)
        cal_time.append(tmp)
    cal_times.append(cal_time)

# Verificar o tamanho das listas antes de acessar os índices
for i in range(USER):
    if len(cal_times[i]) > USER:
        print(f"client {i+1} processing time: ", cal_times[i][USER])
    else:
        print(f"Erro: client {i+1} não tem {USER} elementos em cal_times")

# Verificar o tamanho das listas antes de acessar os índices
for i in range(USER):
    if len(val_acc[i]) > 0:
        print(f"Max acc of client {i+1}: ", max(val_acc[i]))
    else:
        print(f"Erro: client {i+1} não tem elementos em val_acc")

# Gráfico de barras para Epoch vs Accuracy
plt.figure()
for i in range(USER):
    if len(val_acc[i]) > 0:
        plt.bar(range(len(val_acc[i])), val_acc[i], label=f'Client {i+1}')
        for x, y in zip(range(len(val_acc[i])), val_acc[i]):
            plt.annotate(f'{y:.2f}', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.legend(fontsize=14)
plt.title('Epoch vs Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.tick_params(labelsize=14)
plt.grid()
plt.subplots_adjust(left=0.140, right=0.980, bottom=0.130, top=0.870)
plt.savefig('./images/figure5.png')

# Gráfico de barras horizontais para Processing time [s] vs Accuracy
plt.figure()
for i in range(USER):
    if len(cal_times[i]) > 0 and len(val_acc[i]) > 0:
        plt.barh(cal_times[i], val_acc[i], label=f'Client {i+1}')
        for x, y in zip(cal_times[i], val_acc[i]):
            plt.annotate(f'{y:.2f}', (y, x), textcoords="offset points", xytext=(10,0), ha='center')

plt.legend(fontsize=14)
plt.title('Processing time [s] vs Accuracy', fontsize=14)
plt.xlabel('Accuracy', fontsize=14)
plt.ylabel('Processing time [s]', fontsize=14)
plt.tick_params(labelsize=14)
plt.grid()
plt.subplots_adjust(left=0.140, right=0.980, bottom=0.130, top=0.870)
plt.savefig('./images/figure6.png')

