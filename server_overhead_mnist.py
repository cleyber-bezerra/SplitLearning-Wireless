'''
Split learning (Client A -> Server -> Client A)
Server program
'''

import os
import socket
import pickle
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import time
import sys
import subprocess
import pandas as pd
from torch.autograd import Variable
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from time import process_time

# Adiciona o caminho da pasta onde o arquivo está localizado MyNet
file_path = os.path.abspath(os.path.join('.', 'nets'))
sys.path.append(file_path)

import MyNet
import socket_fun as sf
DAM = b'ok!'   # dammy Send credit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

mymodel = MyNet.MyNet_hidden().to(device)
print("mymodel: ", mymodel)

### DEFINIR OS CAMINHOS DE PASTAS
pastas = ["./results/csv/ia", "./results/csv/ns3", "./results/img"]
# Verificar e criar as pastas se não existirem
for pasta in pastas:
    if not os.path.exists(pasta):
        os.makedirs(pasta)
        print(f"Pasta '{pasta}' criada.")


# -------------------- connection ----------------------
# connection establish
user_info = []
host = '127.0.0.1'
port = 19089
ADDR = (host, port)
s = socket.socket()
s.bind(ADDR)
USER = 1
s.listen(USER)
print("Waiting clients...")

# OPEN CLIENTS
# BUSCAR ARQUIVO CSV COM DADOS DE REDE
file_path_ns3 = './results/csv/ns3/simulator_ns3.csv'
# Verificar se o arquivo existe
if os.path.isfile(file_path_ns3):
    # Carregar o arquivo CSV com a sequência dos clientes e os dados de delay
    df = pd.read_csv('./results/csv/ns3/simulator_ns3.csv')
else:
    print(f"O arquivo com dados de rede (.csv) não foi encontrado. Deve ser gerado no ns3")

# Definir a sequência dos clientes e o vetor de accuracy_entrada
client_sequence = df['Flow ID'].tolist()
accuracy_entrada = df['Latency (s)'].tolist()

FLAG = 1 # 0 - todos com latência, 1 - filtro de latencias
if FLAG == 1 :
    # Filtrar clientes onde accuracy_entrada é menor ou igual a 0.4 e diferente de 'inf'
    filtered_clients = [client_sequence[i] for i in range(len(accuracy_entrada)) if accuracy_entrada[i] <= 0.40 and accuracy_entrada[i] != float('inf')]
    clients = client_sequence
else:
    filtered_clients = [client_sequence[i] for i in range(len(accuracy_entrada)) if accuracy_entrada[i] != float('inf')]
    print('Clients in the training set: {filtered_clients}')
    clients = filtered_clients

print(clients)

# Caminho para o interpretador Python
python_interpreter = "python3"

ASYNC = 0 # 0 - modelo sincrono, 1 - assincrono

if ASYNC==1:
    # Executar cada script em um novo terminal
    for client in clients:
        script_path = f"./clients/client{client}_async.py"
        subprocess.Popen(['gnome-terminal', '--', python_interpreter, script_path])
else:
    # Executar cada script em um novo terminal
    for client in clients:
        script_path = f"./clients/client{client}_sync.py"
        subprocess.Popen(['gnome-terminal', '--', python_interpreter, script_path])

# CONNECT
for num_user in range(USER):
    conn, addr = s.accept()
    user_info.append({"name": "Client "+str(num_user+1), "conn": conn, "addr": addr})
    print("Connected with Client "+str(num_user+1), addr)

# RECEIVE
for user in user_info:
    recvreq = user["conn"].recv(1024)
    print("receive request message from client <{}>".format(user["addr"]))
    user["conn"].sendall(DAM)   # send dammy

# ------------------- start training --------------------
def train(user):
    # Variáveis para contabilizar a sobrecarga de comunicação
    total_communication_overhead = 0
    communication_overhead_per_second = []

    # store the time training starts
    p_start = process_time()
    start_time = time.time()

    i = 1
    ite_counter = -1
    user_counter = 0
    lr = 0.005
    optimizer = torch.optim.SGD(mymodel.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    LOADFLAG = 0

    while True:
        ### receive MODE
        recv_mode = sf.recv_size_n_msg(user["conn"])

        # ============= train mode ============
        if recv_mode == 0:
            mymodel.train()
            ite_counter += 1
            print("(USER {}) TRAIN Loading... {}".format(i, ite_counter))

            # RECEIVE ---------- feature data 1 ---------
            recv_data1 = sf.recv_size_n_msg(user["conn"])
            total_communication_overhead += sys.getsizeof(recv_data1)

            # Reset optimization function
            optimizer.zero_grad()

            # Forward prop. 2
            output_2 = mymodel(recv_data1)

            # SEND ------------ feature data 2 ----------
            sf.send_size_n_msg(output_2, user["conn"])
            total_communication_overhead += sys.getsizeof(output_2)

            # RECEIVE ------------ grad 2 ------------
            recv_grad = sf.recv_size_n_msg(user["conn"])
            total_communication_overhead += sys.getsizeof(recv_grad)

            # Back prop.
            output_2.backward(recv_grad)

            # update param.
            optimizer.step()

            # SEND ------------- grad 1 -----------
            sf.send_size_n_msg(recv_data1.grad, user["conn"])
            total_communication_overhead += sys.getsizeof(recv_data1.grad)

        # ============= test mode =============
        elif recv_mode == 1:
            ite_counter = -1
            mymodel.eval()
            print("(USER {}) TEST Loading...".format(i))

            # RECEIVE ---------- feature data 1 -----------
            recv_data = sf.recv_size_n_msg(user["conn"])
            total_communication_overhead += sys.getsizeof(recv_data)

            output_2 = mymodel(recv_data)

            # SEND ---------- feature data 2 ------------
            sf.send_size_n_msg(output_2, user["conn"])
            total_communication_overhead += sys.getsizeof(output_2)

        # =============== move to the next client =============
        elif recv_mode == 2:
            ite_counter = -1
            print(user["name"], " finished training!!!")
            i = i % USER
            print("Now user ", i+1, "is")
            user = user_info[i]
            i += 1

        # ============== this client done, move to the next client ==========
        elif recv_mode == 3:
            user_counter += 1
            i = i % USER
            print(user["name"], "all done!!!!")
            user["conn"].close()
            if user_counter == USER: break
            user = user_info[i]
            i += 1

        else:
            print("!!!!! MODE error !!!!!")

        # Registrar a sobrecarga de comunicação por segundo
        current_time = time.time()
        if current_time - start_time >= 1:
            communication_overhead_per_second.append(total_communication_overhead)
            total_communication_overhead = 0
            start_time = current_time

    print("=============Training is done!!!!!!===========")
    print("Finished the socket connection(SERVER)")

    # store the time training ends
    p_finish = process_time()
    print("Processing time: ", p_finish - p_start)

    # Salvar a sobrecarga de comunicação por segundo em um arquivo
    with open('communication_overhead.pkl', 'wb') as f:
        pickle.dump(communication_overhead_per_second, f)

    # Executar plotagens
    plot_acc_file_path = f"./plots/plot_ai_result3.py"
    subprocess.Popen(['gnome-terminal', '--', python_interpreter, plot_acc_file_path])
    plot_acc_file_path = f"./plots/plot_net_result.py"
    subprocess.Popen(['gnome-terminal', '--', python_interpreter, plot_acc_file_path])
    plot_acc_file_path = f"./plots/plot_net_group.py"
    subprocess.Popen(['gnome-terminal', '--', python_interpreter, plot_acc_file_path])

if __name__ == '__main__':
    train(user_info[0])
