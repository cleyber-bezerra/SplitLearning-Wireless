from matplotlib import pyplot as plt
import csv
import pandas as pd

train_loss, train_acc, val_loss, val_acc = [], [], [], []
p_time = []
cal_times = []

# Caminho para o arquivo CSV
file = './csv/ia/result_train_async.csv'

# Abrir o arquivo e contar as linhas
with open(file, 'r') as f:
    USER = sum(1 for line in f)

# Inicialize as listas antes de usá-las
train_loss = []
train_acc = []
val_loss = []
val_acc = []
p_time = []

def make_list_from_csv(file, USER):
    with open(file, newline='') as f:
        csvreader = csv.reader(f)
        content = [row for row in csvreader]  # [ [row],[row],[row],[row] ]
        
        # Verificar se o número de linhas no CSV é suficiente
        if len(content) < USER:
            raise ValueError("O arquivo CSV não tem linhas suficientes para o valor de USER fornecido.")
        
        for i, row in enumerate(content):
            if i >= USER:
                break
            try:
                train_loss.append(row[1])
                train_acc.append(row[2])
                val_loss.append(row[3])
                val_acc.append(row[4])
                p_time.append(row[5])
            except IndexError:
                print(f"Erro ao acessar índices na linha {i+1}. Certifique-se de que todas as linhas tenham o número correto de colunas.")
                break

make_list_from_csv(file, USER)

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

plt.figure()
for i in range(USER):
    if len(cal_times[i]) > 0 and len(val_acc[i]) > 0:
        plt.plot(cal_times[i], val_acc[i], linewidth=2, linestyle='-', label=f'Client {i+1}')
        for x, y in zip(cal_times[i], val_acc[i]):
            plt.annotate(f'({x:.1f}, {y:.2f})', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.legend(fontsize=14)
plt.title('Processing time [s] vs Accuracy', fontsize=14)
plt.xlabel('Processing time [s]', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.tick_params(labelsize=14)
plt.xlim(0,)
plt.ylim(0,)
plt.grid()
plt.subplots_adjust(left=0.140, right=0.980, bottom=0.130, top=0.870)
plt.savefig('./images/figure6a.png')

plt.figure()
for i in range(USER):
    if len(val_acc[i]) > 0:
        plt.plot(range(len(val_acc[i])), val_acc[i], linewidth=2, linestyle='-', label=f'Client {i+1}')
        for x, y in zip(range(len(val_acc[i])), val_acc[i]):
            plt.annotate(f'({x}, {y:.2f})', (x, y), textcoords="offset points", xytext=(0,10), ha='center')

plt.legend(fontsize=14)
plt.title('Epoch vs Accuracy', fontsize=14)
plt.xlabel('Epoch', fontsize=14)
plt.ylabel('Accuracy', fontsize=14)
plt.tick_params(labelsize=14)
plt.xlim(0,)
plt.ylim(0,)
plt.grid()
plt.subplots_adjust(left=0.140, right=0.980, bottom=0.130, top=0.870)
plt.savefig('./images/figure7a.png')

