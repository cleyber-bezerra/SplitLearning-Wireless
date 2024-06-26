''' 
Split learning (Client A -> Server -> Client A)
Server program
'''

from email.generator import BytesGenerator
import os
from pyexpat import model
import socket
import struct
import pickle

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import sys
import copy
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

#calls clientes
import subprocess
import pandas as pd

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
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
# -------------------- conexão ----------------------
# connection establish
# conexão estabelecida
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


# Executar cada script em um novo terminal
for client in clients:
    script_path = f"./clients/client{client}.py"
    subprocess.Popen(['gnome-terminal', '--', python_interpreter, script_path])
    # Ou se você usar xterm:
    #subprocess.Popen(['xterm', '-e', python_interpreter + ' ' + script_path])

# CONNECT
# CONECTAR
for num_user in range(USER):
    conn, addr = s.accept()
    user_info.append({"name": "Client "+str(num_user+1), "conn": conn, "addr": addr})
    print("Connected with Client "+str(num_user+1), addr)

# RECEIVE
# RECEBER
for user in user_info:
    recvreq = user["conn"].recv(1024)
    print("receive request message from client <{}>".format(user["addr"]))
    user["conn"].sendall(DAM)   # send dammy




# ------------------- start training --------------------
# ------------------- comece a treinar --------------------
def train(user):

    # store the time training starts
    # armazena o horário de início do treinamento
    p_start = process_time()

    i = 1
    ite_counter = -1
    user_counter = 0
    # PATH = []
    # PATH.append('./savemodels/client1.pth')
    # PATH.append('./savemodels/client2.pth')
    # PATH.append('./savemodels/client3.pth')
    lr = 0.005
    optimizer = torch.optim.SGD(mymodel.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
    LOADFLAG = 0

    while True:
        ### receive MODE
        ### modo de recepção
        recv_mode = sf.recv_size_n_msg(user["conn"])

        # ============= train mode ============
        # ============= modo trem ============
        if recv_mode == 0:
            mymodel.train()
            # if LOADFLAG == 1:
            #     mymodel.load_state_dict(torch.load(PATH[user_counter-1]))
            #     LOADFLAG = 0

            ite_counter+=1
            print("(USER {}) TRAIN Loading... {}".format(i, ite_counter))

            # RECEIVE ---------- feature data 1 ---------
            # RECEBER ---------- dados do recurso 1 ---------
            recv_data1 = sf.recv_size_n_msg(user["conn"])

            # Reset optimization function
            optimizer.zero_grad()

            # Forward prop. 2
            # Suporte para frente. 2
            output_2 = mymodel(recv_data1)

            # SEND ------------ feature data 2 ----------
            # ENVIAR ------------ dados do recurso 2 ----------
            sf.send_size_n_msg(output_2, user["conn"])

            # ==================== Forward propagation completed =====================

            # RECEIVE ------------ grad 2 ------------
            # RECEBER ------------ 2ª série ------------
            recv_grad = sf.recv_size_n_msg(user["conn"])

            # Back prop.
            # Suporte traseiro.
            output_2.backward(recv_grad)

            # update param.
            # atualiza parâmetro.
            optimizer.step()

            # SEND ------------- grad 1 -----------
            # ENVIAR ------------- 1ª série -----------
            sf.send_size_n_msg(recv_data1.grad, user["conn"])

        # ============= test mode =============
        # ============= modo de teste =============
        elif recv_mode == 1:
            ite_counter = -1
            mymodel.eval()
            print("(USER {}) TEST Loading...".format(i))

            # RECEIVE ---------- feature data 1 -----------
            # RECEIVE ---------- dados do recurso 1 -----------
            recv_data = sf.recv_size_n_msg(user["conn"])

            output_2 = mymodel(recv_data)

            # SEND ---------- feature data 2 ------------
            # ENVIAR ---------- dados do recurso 2 ------------
            sf.send_size_n_msg(output_2, user["conn"])

        # =============== move to the next client =============
        # =============== passar para o próximo cliente =============
        elif recv_mode == 2:    # Epoch EACH verの場合
            ite_counter = -1
            # torch.save(mymodel.state_dict(), PATH[user_counter-1])
            # LOADFLAG = 1
            print(user["name"], " finished training!!!")
            i = i%USER
            print("Now user ", i+1, "is")
            user = user_info[i]
            i += 1
        
        # ============== this client done, move to the next client ==========
        # ============== este cliente terminou, vá para o próximo cliente ==========
        elif recv_mode == 3:
            user_counter += 1
            i = i%USER
            # torch.save(mymodel.state_dict(), PATH[user_counter-1])
            # LOADFLAG = 1
            print(user["name"], "all done!!!!")
            user["conn"].close()
            if user_counter == USER: break
            user = user_info[i]
            i += 1

        else:   print("!!!!! MODE error !!!!!")

    print("=============Training is done!!!!!!===========")
    print("Finished the socket connection(SERVER)")

    # store the time training ends
    # armazena o horário em que o treinamento termina
    p_finish = process_time()

    print("Processing time: ",p_finish-p_start)

    #Executar plotagens
     #gráficos de ML(machine learning)
    plot_acc_file_path = f"./plots/plot_ai_result3.py"
    subprocess.Popen(['gnome-terminal', '--', python_interpreter, plot_acc_file_path])
     #gráficos de Net(Simulator NS3)
    plot_acc_file_path = f"./plots/plot_net_result.py"
    subprocess.Popen(['gnome-terminal', '--', python_interpreter, plot_acc_file_path])
    plot_acc_file_path = f"./plots/plot_net_group.py"
    subprocess.Popen(['gnome-terminal', '--', python_interpreter, plot_acc_file_path])

if __name__ == '__main__':
    train(user_info[0])