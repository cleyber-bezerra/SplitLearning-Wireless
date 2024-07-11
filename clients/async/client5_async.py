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
import asyncio

# Adiciona o caminho da pasta onde o arquivo está localizado ml_model
file_path = os.path.abspath(os.path.join('.', 'models'))
sys.path.append(file_path)

import ml_model
import socket_fun as sf

### Variáveis globais
DAM = b'ok!'    # dammy
MODE = 0    # 0->train, 1->test
BATCH_SIZE = 128

print(" ------ CLIENT 5 ------")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

# Argumento de linha de comando para delay
if len(sys.argv) < 2:
    print("Uso: client_async.py <delay>")
    sys.exit(1)

delay = float(sys.argv[1])
print(f"Accuracy recebido: {delay}")

#MNIST
root = './datasets/mnist_data'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Parâmetros de normalização para MNIST
])

# Download dataset
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

# -------------------- Conexão ----------------------
host = '127.0.0.1'
port = 19090
ADDR = (host, port)

# Conectar ao servidor
s = socket.socket()
s.connect(ADDR)
w = 'start communication'

# Enviar solicitação ao servidor
s.sendall(w.encode())
print("Solicitação enviada ao servidor")
dammy = s.recv(4)

# ------------------ Iniciar treinamento -----------------
epochs = 1
lr = 0.005

# Definir função de erro
criterion = nn.CrossEntropyLoss()

# Definir otimizadores
optimizer1 = optim.SGD(mymodel1.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)
optimizer2 = optim.SGD(mymodel2.parameters(), lr=lr, momentum=0.9, weight_decay=5e-4)

# Variáveis para contabilizar a sobrecarga de comunicação
comm_time = 0
comm_data_size = 0

# Função para quantificar dados
def quantize(tensor, num_bits=8):
    scale = 2 ** num_bits - 1
    tensor = torch.clamp(tensor, 0, 1) * scale
    tensor = torch.round(tensor) / scale
    return tensor

async def train():
    global comm_time, comm_data_size

    # Função de propagação para frente
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
        mymodel1.train()
        mymodel2.train()
        MODE = 0    # Modo treino -> 0:train, 1:test, 2:finished train, 3:finished test
        MODEL = 1   # Modelo de treino
        for data, labels in tqdm(trainloader):
            sf.send_size_n_msg(MODE, s)
        
            data = data.to(device)
            labels = labels.to(device)

            output_1 = forward_prop(MODEL, data)

            time.sleep(delay)
            start_time = process_time()
            quantized_output_1 = quantize(output_1)
            sf.send_size_n_msg(quantized_output_1, s)
            comm_time += process_time() - start_time
            comm_data_size += quantized_output_1.element_size() * quantized_output_1.nelement()

            recv_data2 = sf.recv_size_n_msg(s)
            start_time = process_time()
            comm_time += process_time() - start_time
            comm_data_size += recv_data2.element_size() * recv_data2.nelement()

            MODEL = 2
            OUTPUT = forward_prop(MODEL, recv_data2)
            loss = criterion(OUTPUT, labels)

            loss.backward()     # Peças da camada externa
            optimizer2.step()

            start_time = process_time()
            sf.send_size_n_msg(recv_data2.grad, s)
            comm_time += process_time() - start_time
            comm_data_size += recv_data2.grad.element_size() * recv_data2.grad.nelement()

            time.sleep(delay)
            start_time = process_time()
            recv_grad = sf.recv_size_n_msg(s)
            comm_time += process_time() - start_time
            comm_data_size += recv_grad.element_size() * recv_grad.nelement()

            MODEL = 1
            train_loss = train_loss + loss.item()
            train_acc += (OUTPUT.max(1)[1] == labels).sum().item()

            output_1.backward(recv_grad)    # Peças em camada
            time.sleep(0.3521)  # Adiciona latência no gradiente

            optimizer1.step()
        
        avg_train_loss = train_loss / len(trainloader.dataset)
        avg_train_acc = train_acc / len(trainloader.dataset)
        
        print("Modo treino finalizado!!!!")
        
        train_loss_list.append(avg_train_loss)
        train_acc_list.append(avg_train_acc)

        mymodel1.eval()
        mymodel2.eval()

        with torch.no_grad():
            print("Iniciando modo de teste!")
            MODE = 1
            for data, labels in tqdm(testloader):
                sf.send_size_n_msg(MODE, s)

                data = data.to(device)
                labels = labels.to(device)
                output = mymodel1(data)

                start_time = process_time()
                quantized_output = quantize(output)
                sf.send_size_n_msg(quantized_output, s)
                comm_time += process_time() - start_time
                comm_data_size += quantized_output.element_size() * quantized_output.nelement()

                recv_data2 = sf.recv_size_n_msg(s)
                start_time = process_time()
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
            MODE = 2
        sf.send_size_n_msg(MODE, s)
        print("Tempo de processamento: ", p_time)
        p_time_list.append(p_time)

        avg_val_loss = val_loss / len(testloader.dataset)
        avg_val_acc = val_acc / len(testloader.dataset)
        
        print('Epoch [{}/{}], Loss: {:.5f}, Acc: {:.5f},  val_loss: {:.5f}, val_acc: {:.5f}' 
              .format(e+1, epochs, avg_train_loss, avg_train_acc, avg_val_loss, avg_val_acc))
        
        val_loss_list.append(avg_val_loss)
        val_acc_list.append(avg_val_acc)

        if e == epochs-1: 
            s.close()
            print("Conexão de soquete finalizada (CLIENTE 5)")

    return train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list, comm_time, comm_data_size

import csv

def write_to_csv(train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list, comm_time, comm_data_size):
    file = './csv/ia/result_train_async.csv'
    file_exists = os.path.isfile(file)
    with open(file, 'a', newline='') as f:
        csv_writer = csv.writer(f)

        if not file_exists:
            csv_writer.writerow(['Client', 'Train Loss', 'Train Accuracy', 'Validation Loss', 'Validation Accuracy', 'Processing Time', 'Comm Time', 'Comm Data Size', 'Timestamp'])

        timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

        result = [
            'client 5',
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
    train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list, comm_time, comm_data_size = asyncio.run(train())
    write_to_csv(train_loss_list, train_acc_list, val_loss_list, val_acc_list, p_time_list, comm_time, comm_data_size)
