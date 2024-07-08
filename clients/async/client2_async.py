'''
Shared Split learning (Client 1 -> Server -> Client 1)
Client 1 program
'''
import gc
from glob import escape
import os
from pyexpat import model
import socket
import struct
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

# Adiciona o caminho da pasta onde o arquivo está localizado ml_model
file_path = os.path.abspath(os.path.join('.', 'models'))
sys.path.append(file_path)

import ml_model
import socket_fun as sf
from asynchronous import AsynchronousSplitLearning  # Importe a classe do novo arquivo

### global variable
### variável global
DAM = b'ok!'    # dammy
MODE = 0    # 0->train, 1->test

BATCH_SIZE = 128

print(" ------ CLIENT 2 - ASYNC ------")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# device = torch.device('cpu')
print("device: ", device)

#MNIST
root = './datasets/mnist_data'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Parâmetros de normalização para MNIST
])

# download dataset
trainset = torchvision.datasets.MNIST(root=root, download=True, train=True, transform=transform)
# if you want to divide dataset, remove comments below.
# se você deseja dividir o conjunto de dados, remova os comentários abaixo.
# indices = np.arange(len(trainset))
# train_dataset = torch.utils.data.Subset(
#     trainset, indices[0:20000]
# )
# trainset = train_dataset
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

    asl = AsynchronousSplitLearning([mymodel1], mymodel2, num_epoch=epochs, num_batch=len(trainloader), K=1, lthred=0.01)

    train_loss_list, train_acc_list, val_loss_list, val_acc_list = [], [], [], []
    p_time_list = []

    latencies = [0.1, 0.5, 1.0]  # Exemplos de latências
    delta_t = 0.2  # Threshold para latência
    error_rate = 0.1  # Taxa de erro simulada

    for e in range(epochs):
        print("--------------- Epoch: ", e, " --------------")
        p_start = process_time()

        mymodel1.train()
        mymodel2.train()
        asl.train(trainloader, criterion, optimizer1, latencies, delta_t, error_rate, device)
        
        mymodel1.eval()
        mymodel2.eval()

        train_loss = 0
        train_acc = 0
        val_loss = 0
        val_acc = 0

        with torch.no_grad():
            print("start test mode!")
            for data, labels in tqdm(testloader):
                data = data.to(device)
                labels = labels.to(device)
                output = mymodel1(data)

                # SEND --------- feature data 1 -----------
                start_time = process_time()
                sf.send_size_n_msg(output, s)
                comm_time += process_time() - start_time
                comm_data_size += output.element_size() * output.nelement()

                ### wait for the server...
                # RECEIVE ----------- feature data 2 ------------
                start_time = process_time()
                recv_data2 = sf.recv_size_n_msg(s)
                comm_time += process_time() - start_time
                comm_data_size += recv_data2.element_size() * recv_data2.nelement()

                OUTPUT = mymodel2(recv_data2)
                loss = criterion(OUTPUT, labels)

                val_loss += loss.item()
                val_acc += (OUTPUT.max(1)[1] == labels).sum().item()

        p_finish = process_time()
        p_time = p_finish - p_start
        if e == epochs - 1:
            MODE = 3
        else:
            MODE = 2
        sf.send_size_n_msg(MODE, s)
        print("Processing time: ", p_time)
        p_time_list.append(p_time)

        avg_train_loss = train_loss / len(trainloader.dataset)
        avg_train_acc = train_acc / len(trainloader.dataset)
        avg_val_loss = val_loss / len(testloader.dataset)
        avg_val_acc = val_acc / len(testloader.dataset)

        print('Epoch [{}/{}], Loss: {:.5f}, Acc: {:.5f}, val_loss: {:.5f}, val_acc: {:.5f}'
              .format(e + 1, epochs, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc))

        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

        if e == epochs - 1:
            s.close()
            print("Finished the socket connection(CLIENT 3)")

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list, comm_time, comm_data_size

import csv

def write_to_csv(train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list, comm_time, comm_data_size):
    file = './csv/ia/result_train_async.csv'
    f = open(file, 'a', newline='')
    csv_writer = csv.writer(f)

    result = []
    result.append('client 3')
    result.append(train_loss_list)
    result.append(train_acc_list)
    result.append(val_loss_list)
    result.append(val_acc_list)
    result.append(p_time_list)
    result.append(comm_time)
    result.append(comm_data_size)

    csv_writer.writerow(result)

    f.close()

if __name__ == '__main__':
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list, comm_time, comm_data_size = train()
    write_to_csv(train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list, comm_time, comm_data_size)

