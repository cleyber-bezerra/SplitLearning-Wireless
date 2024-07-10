import gc
import os
import socket
import pickle
import numpy as np
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
from datetime import datetime

# Adiciona o caminho da pasta onde o arquivo está localizado ml_model
file_path = os.path.abspath(os.path.join('.', 'models'))
sys.path.append(file_path)

import ml_model
import socket_fun as sf

### global variable
### variável global
DAM = b'ok!'    # dammy
MODE = 0    # 0->train, 1->test

BATCH_SIZE = 128

print(" ------ CLIENT 1 ------")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

# Argumento de linha de comando para accuracy
if len(sys.argv) < 2:
    print("Uso: client_sync.py <accuracy>")
    sys.exit(1)

accuracy = float(sys.argv[1])
print(f"Accuracy recebido: {accuracy}")

#MNIST
root = './datasets/mnist_data'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Parâmetros de normalização para MNIST
])

# download dataset
trainset = torchvision.datasets.MNIST(root=root, download=True, train=True, transform=transform)
testset = torchvision.datasets.MNIST(root=root, download=True, train=False, transform=transform)

print("trainset_len: ", len(trainset))
print("testset_len: ", len(testset))
image, label = trainset[0]
print (image.size())

trainloader = DataLoader(trainset, batch_size=BATCH_SIZE, shuffle=True, num_workers=2)
testloader = DataLoader(testset, batch_size=BATCH_SIZE, shuffle=False, num_workers=2)

mymodel1 = ml_model.ml_model_in().to(device)
mymodel2 = ml_model.ml_model_out(NUM_CLASSES=10).to(device)

# -------------------- connection ----------------------
# -------------------- conexão ----------------------
# socket establish
# estabelecimento de soquete
host = '127.0.0.1'
port = 19089
ADDR = (host, port)

# CONNECT
# CONECTAR
s = socket.socket()
s.connect(ADDR)
w = 'start communication'

# SEND
# ENVIAR
s.sendall(w.encode())
print("sent request to a server")
dammy = s.recv(4)

# ------------------ start training -----------------
# ------------------ comece a treinar -----------------
epochs = 1
lr = 0.005

# set error function
# define função de erro
criterion = nn.CrossEntropyLoss()

# set optimizer
# definir otimizador
optimizer1 = optim.SGD(mymodel1.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(mymodel2.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# Variáveis para contabilizar a sobrecarga de comunicação
comm_time = 0
comm_data_size = 0

def train():
    global comm_time, comm_data_size

    # forward prop. function
    # suporte para frente. função
    def forward_prop(MODEL, data):

        output = None
        if MODEL == 1:
            optimizer1.zero_grad()
            output_1 = mymodel1(data)
            time.sleep(0.3521)  # Adiciona latência na ativação
            output = output_1
        elif MODEL == 2:
            optimizer2.zero_grad()
            output_2 = mymodel2(data)
            time.sleep(0.3521)  # Adiciona latência na ativação
            output = output_2
        else:
            print("!!!!! MODEL not found !!!!!")

        return output

    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    p_time_list = []

    for e in range(epochs):
        print("--------------- Epoch: ", e, " --------------")
        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        p_start = process_time()
        # ================= train mode ================
        # ================= modo trem ================
        mymodel1.train()
        mymodel2.train()
        ### send MODE
        ### enviar MODO
        MODE = 0    # train mode -> 0:train, 1:test, 2:finished train, 3:finished test
        MODEL = 1   # train model
        for data, labels in tqdm(trainloader):
            # send MODE number
            # envia o número do MODO
            sf.send_size_n_msg(MODE, s)
        
            data = data.to(device)
            labels = labels.to(device)

            output_1 = forward_prop(MODEL, data)

            time.sleep(accuracy)
            # SEND ----------- feature data 1 ----------------
            # ENVIAR ----------- dados do recurso 1 ---------------
            start_time = process_time()
            sf.send_size_n_msg(output_1, s)
            comm_time += process_time() - start_time
            comm_data_size += output_1.element_size() * output_1.nelement()

            ### wait for SERVER to calculate... ###
            ### espere o SERVIDOR calcular... ###

            # RECEIVE ------------ feature data 2 -------------
            # RECEBER ------------ dados do recurso 2 -------------
            start_time = process_time()
            recv_data2 = sf.recv_size_n_msg(s)
            comm_time += process_time() - start_time
            comm_data_size += recv_data2.element_size() * recv_data2.nelement()

            # recebe dados do recurso 2 -> MODEL=2
            MODEL = 2   # receive feature data 2 -> MODEL=2

            # start forward prop. 3
            # inicia a hélice para frente. 3
            OUTPUT = forward_prop(MODEL, recv_data2)

            loss = criterion(OUTPUT, labels)

            loss.backward()     # parts out-layer (peças da camada externa)

            optimizer2.step()

            # SEND ------------- grad 2 -----------
            # ENVIAR ------------- 2ª série -----------
            start_time = process_time()
            sf.send_size_n_msg(recv_data2.grad, s)
            comm_time += process_time() - start_time
            comm_data_size += recv_data2.grad.element_size() * recv_data2.grad.nelement()

            time.sleep(accuracy)
            # RECEIVE ----------- grad 1 -----------
            # RECEBER ----------- 1ª série -----------
            start_time = process_time()
            recv_grad = sf.recv_size_n_msg(s)
            comm_time += process_time() - start_time
            comm_data_size += recv_grad.element_size() * recv_grad.nelement()

            MODEL = 1

            train_loss = train_loss + loss.item()
            train_acc += (OUTPUT.max(1)[1] == labels).sum().item()

            output_1.backward(recv_grad)    # parts in-layer (peças em camada)
            time.sleep(0.3521)  # Adiciona latência no gradiente

            optimizer1.step()
        
        avg_train_loss = train_loss / len(trainloader.dataset)
        avg_train_acc = train_acc / len(trainloader.dataset)
        
        print("train mode finished!!!!")
        
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)

        # =============== test mode ================
        # =============== modo de teste ================
        mymodel1.eval()
        mymodel2.eval()

        with torch.no_grad():
            print("start test mode!")
            MODE = 1    # change mode to test (mude o modo para testar)
            for data, labels in tqdm(testloader):
                # send MODE number
                # envia o número do MODO
                sf.send_size_n_msg(MODE, s)

                data = data.to(device)
                labels = labels.to(device)
                output = mymodel1(data)

                # SEND --------- feature data 1 -----------
                # ENVIAR --------- dados do recurso 1 -----------
                start_time = process_time()
                sf.send_size_n_msg(output, s)
                comm_time += process_time() - start_time
                comm_data_size += output.element_size() * output.nelement()

                ### wait for the server...
                ### espere pelo servidor...

                # RECEIVE ----------- feature data 2 ------------
                # RECEBER ----------- dados do recurso 2 ------------
                start_time = process_time()
                recv_data2 = sf.recv_size_n_msg(s)
                comm_time += process_time() - start_time
                comm_data_size += recv_data2.element_size() * recv_data2.nelement()

                OUTPUT = mymodel2(recv_data2)
                loss = criterion(OUTPUT, labels)

                val_loss += loss.item()
                val_acc += (OUTPUT.max(1)[1] == labels).sum().item()

        p_finish = process_time()
        p_time = p_finish-p_start
        if e == epochs-1:
            MODE = 3
        else:                                 
            MODE = 2    # finished test -> start next client's training
        sf.send_size_n_msg(MODE, s)           
        print("Processing time: ", p_time)
        p_time_list.append(p_time)

        avg_val_loss = val_loss / len(testloader.dataset)
        avg_val_acc = val_acc / len(testloader.dataset)
        
        print ('Epoch [{}/{}], Loss: {loss:.5f}, Acc: {acc:.5f},  val_loss: {val_loss:.5f}, val_acc: {val_acc:.5f}' 
                    .format(e+1, epochs, loss=avg_train_loss, acc=avg_train_acc, val_loss=avg_val_loss, val_acc=avg_val_acc))
        
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

        if e == epochs-1: 
            s.close()
            print("Finished the socket connection(CLIENT 1)")

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list, comm_time, comm_data_size

import csv

def write_to_csv(train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list, comm_time, comm_data_size):
    file = './csv/ia/result_train_sync.csv'
    file_exists = os.path.isfile(file)
    with open(file, 'a', newline='') as f:
        csv_writer = csv.writer(f)

        if not file_exists:
            # Se o arquivo não existe, escrever o cabeçalho
            csv_writer.writerow(['Client', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy', 'Processing Time', 'Comm Time', 'Comm Data Size', 'Timestamp'])

        # Adiciona a data e hora atual
        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        result = [
            'client 1',
            train_loss_list,
            train_acc_list,
            val_loss_list,
            val_acc_list,
            p_time_list,
            comm_time,
            comm_data_size,
            timestamp
        ]

        csv_writer.writerow(result)

if __name__ == '__main__':
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list, comm_time, comm_data_size = train()
    write_to_csv(train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list, comm_time, comm_data_size)
