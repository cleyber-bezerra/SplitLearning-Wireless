import os
import socket
import pickle
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import torch.optim as optim
import time
import sys
import copy
from torch.utils.data import DataLoader
from time import process_time
from datetime import datetime
import csv

import ml_model
import socket_fun as sf

DAM = b'ok!'  # Dammy
MODE = 0  # 0->train, 1->test

BATCH_SIZE = 128

print(" ------ CLIENT 1 ------")
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print("device: ", device)

# Argumento de linha de comando para delay
if len(sys.argv) < 2:
    print("Uso: client_sync.py <delay>")
    sys.exit(1)

delay = float(sys.argv[1])
print(f"Delay recebido: {delay}")

# MNIST
root = './datasets/mnist_data'
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.1307,), (0.3081,))  # Parâmetros de normalização do MNIST
])
dataset = torchvision.datasets.MNIST(root=root, train=True, transform=transform, download=True)
train_loader = DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)

# Define o modelo
mymodel = ml_model.ml_model_hidden().to(device)
print("mymodel: ", mymodel)

def write_to_csv(result, connection_time):
    now = datetime.now()
    filename = './csv/ia/result_train_sync.csv'
    header = ["Timestamp", "Connection Time (ms)", "Delay (ms)", "Training Time", "Communication Time", "Communication Data"]
    file_exists = os.path.isfile(filename)

    with open(filename, mode='a') as f:
        writer = csv.writer(f)
        if not file_exists:
            writer.writerow(header)
        writer.writerow([now.strftime("%Y-%m-%d %H:%M:%S"), connection_time] + result)

def train():
    # Conectando ao servidor
    host = '127.0.0.1'
    port = 19089
    s = socket.socket()
    s.connect((host, port))

    # Recebe o tempo de conexão e atraso do servidor
    data = s.recv(1024).decode().split(',')
    connection_time = float(data[0])
    server_delay = float(data[1])

    # Aplica o atraso
    time.sleep(server_delay / 1000.0)  # Converte ms para segundos

    # Treinamento
    mymodel.train()
    optimizer = torch.optim.SGD(mymodel.parameters(), lr=0.005, momentum=0.9, weight_decay=5e-4)
    p_start = process_time()

    while True:
        if MODE == 0:
            # Envio de modo de operação
            s.sendall(b'0')

            for batch_idx, (data, target) in enumerate(train_loader):
                start_comm_time = time.time()
                s.sendall(sf.send_size_n_msg(data, s))
                end_comm_time = time.time()
                total_comm_time = (end_comm_time - start_comm_time)
                total_comm_data = sys.getsizeof(data)

                start_comm_time = time.time()
                grad = sf.recv_size_n_msg(s)
                end_comm_time = time.time()
                total_comm_time += (end_comm_time - start_comm_time)
                total_comm_data += sys.getsizeof(grad)

                # Aplicar gradiente
                data.grad = torch.tensor(grad).to(device)
                optimizer.step()

                start_comm_time = time.time()
                s.sendall(sf.send_size_n_msg(data.grad, s))
                end_comm_time = time.time()
                total_comm_time += (end_comm_time - start_comm_time)
                total_comm_data += sys.getsizeof(data.grad)

            # Finaliza o treinamento
            s.sendall(b'2')
            break

        elif MODE == 1:
            s.sendall(b'1')

            for batch_idx, (data, target) in enumerate(train_loader):
                start_comm_time = time.time()
                s.sendall(sf.send_size_n_msg(data, s))
                end_comm_time = time.time()
                total_comm_time = (end_comm_time - start_comm_time)
                total_comm_data = sys.getsizeof(data)

                start_comm_time = time.time()
                output = sf.recv_size_n_msg(s)
                end_comm_time = time.time()
                total_comm_time += (end_comm_time - start_comm_time)
                total_comm_data += sys.getsizeof(output)

        elif MODE == 2:
            break

        elif MODE == 3:
            s.sendall(b'3')
            break

        else:
            print("Modo desconhecido")

    # Finaliza o tempo de treinamento
    p_finish = process_time()
    training_time = p_finish - p_start

    write_to_csv([training_time, total_comm_time, total_comm_data], connection_time)
    s.close()

def main():
    train()

if __name__ == '__main__':
    main()
