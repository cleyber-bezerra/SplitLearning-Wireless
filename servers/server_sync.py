# Imports remain unchanged

import os
import socket
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import torch.optim as optim
import time
import sys
from torch.utils.data import DataLoader
from time import process_time
import subprocess
import pandas as pd

# Adiciona o caminho da pasta onde o arquivo está localizado ml_model
file_path = os.path.abspath(os.path.join('.', 'models'))
sys.path.append(file_path)

import ml_model
import socket_fun as sf
DAM = b'ok!'   # dammy Send credit

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

mymodel = ml_model.ml_model_hidden().to(device)
print("mymodel: ", mymodel)

### DEFINIR OS CAMINHOS DE PASTAS
pastas = ["./csv/ia", "./csv/ns3", "./images"]
# Verificar e criar as pastas se não existirem
for pasta in pastas:
    if not os.path.exists(pasta):
        os.makedirs(pasta)
        print(f"Pasta '{pasta}' criada.")

# -------------------- connection ----------------------
user_info = []
host = '127.0.0.1'
port = 19089
ADDR = (host, port)
s = socket.socket()
s.settimeout(7200) # 7200 seg, 60min, 2h de espera no socket
s.bind(ADDR)
USER = 6 # número de clientes a serem atendidos simultaneamente.
s.listen(USER)
print("Waiting clients...")

# OPEN CLIENTS
df = pd.read_csv('./csv/ns3/simulator_ns3.csv')
client_sequence = df['Flow ID'].tolist()
accuracy_entrada = df['Latency (s)'].tolist()

FLAG = 1 # 0 - todos com latência, 1 - filtro de latencias

if FLAG == 1:
    filtered_clients = [client_sequence[i] for i in range(len(accuracy_entrada)) if accuracy_entrada[i] <= 0.40 and accuracy_entrada[i] != float('inf')]
    clients = client_sequence
else:
    filtered_clients = [client_sequence[i] for i in range(len(accuracy_entrada)) if accuracy_entrada[i] != float('inf')]
    print('Clients in the training set: {filtered_clients}')
    clients = filtered_clients

print(clients)

python_interpreter = "python3"

for client in clients:
    script_path = f"./clients/sync/client{client}_sync.py"
    subprocess.Popen(['gnome-terminal', '--', python_interpreter, script_path])

for num_user in range(USER):
    conn, addr = s.accept()
    user_info.append({"name": "Client "+str(num_user+1), "conn": conn, "addr": addr})
    print("Connected with Client "+str(num_user+1), addr)

for user in user_info:
    recvreq = user["conn"].recv(1024)
    print("receive request message from client <{}>".format(user["addr"]))
    user["conn"].sendall(DAM)

# ------------------- start training --------------------
def train(user):

    p_start = process_time()

    i = 1
    ite_counter = -1
    user_counter = 0
    lr = 0.005
    optimizer = torch.optim.SGD(mymodel.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    LOADFLAG = 0

    # Variáveis para contabilizar a sobrecarga de comunicação
    total_comm_time = 0
    total_comm_data = 0

    while True:
        ### receive MODE
        start_comm_time = time.time()
        recv_mode = sf.recv_size_n_msg(user["conn"])
        end_comm_time = time.time()
        total_comm_time += (end_comm_time - start_comm_time)
        total_comm_data += sys.getsizeof(recv_mode)

        if recv_mode == 0:
            mymodel.train()
            ite_counter += 1
            print("(USER {}) TRAIN Loading... {}".format(i, ite_counter))

            start_comm_time = time.time()
            recv_data1 = sf.recv_size_n_msg(user["conn"])
            end_comm_time = time.time()
            total_comm_time += (end_comm_time - start_comm_time)
            total_comm_data += sys.getsizeof(recv_data1)

            optimizer.zero_grad()

            output_2 = mymodel(recv_data1)

            start_comm_time = time.time()
            sf.send_size_n_msg(output_2, user["conn"])
            end_comm_time = time.time()
            total_comm_time += (end_comm_time - start_comm_time)
            total_comm_data += sys.getsizeof(output_2)

            recv_grad = sf.recv_size_n_msg(user["conn"])

            output_2.backward(recv_grad)

            optimizer.step()

            start_comm_time = time.time()
            sf.send_size_n_msg(recv_data1.grad, user["conn"])
            end_comm_time = time.time()
            total_comm_time += (end_comm_time - start_comm_time)
            total_comm_data += sys.getsizeof(recv_data1.grad)

        elif recv_mode == 1:
            ite_counter = -1
            mymodel.eval()
            print("(USER {}) TEST Loading...".format(i))

            start_comm_time = time.time()
            recv_data = sf.recv_size_n_msg(user["conn"])
            end_comm_time = time.time()
            total_comm_time += (end_comm_time - start_comm_time)
            total_comm_data += sys.getsizeof(recv_data)

            output_2 = mymodel(recv_data)

            start_comm_time = time.time()
            sf.send_size_n_msg(output_2, user["conn"])
            end_comm_time = time.time()
            total_comm_time += (end_comm_time - start_comm_time)
            total_comm_data += sys.getsizeof(output_2)

        elif recv_mode == 2:
            ite_counter = -1
            print(user["name"], " finished training!!!")
            i = i % USER
            print("Now is user ", i + 1)
            user = user_info[i]
            i += 1

        elif recv_mode == 3:
            user_counter += 1
            i = i % USER
            print(user["name"], "all done!!!!")
            user["conn"].close()
            if user_counter == USER:
                break
            user = user_info[i]
            i += 1

        else:
            print("!!!!! MODE error !!!!!")

    print("=============Training is done!!!!!!===========")
    print("Finished the socket connection(SERVER)")

    p_finish = process_time()

    print("Processing time: ", p_finish - p_start)
    print("Total Communication Time: ", total_comm_time)
    print("Total Communication Data: ", total_comm_data, "bytes")

    
    #gráficos de Net(Simulator NS3)
    plot_acc_file_path = f"./plots/plot_net_result.py"
    subprocess.Popen(['gnome-terminal', '--', python_interpreter, plot_acc_file_path])

    # Executar plotagens
    plot_acc_file_path = f"./plots/plot_ai_result_sync.py"
    subprocess.Popen(['gnome-terminal', '--', python_interpreter, plot_acc_file_path])
    
    '''
    plot_acc_file_path = f"./plots/plot_net_group.py"
    subprocess.Popen(['gnome-terminal', '--', python_interpreter, plot_acc_file_path])
    '''

# ACESSO PELO ARQUIVO RUN.PY
def main():
    train(user_info[0])

if __name__ == '__main__':
    main()

