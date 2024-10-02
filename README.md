# SplitLearning-Wireless

<p align='center' style="margin-bottom: -4px">Cleyber Bezerra dos Reis<sup>1</sup>, Antonio Oliveira-JR<sup>1</sup></p>
<p align='center' style="margin-bottom: -4px"><sup>1</sup>Instituto de Informática, Universidade Federal de Goiás</p>
<p align='center' style="margin-bottom: -4px">E-mail: {cleyber.bezerra}@discente.ufg.br</p>
<p align='center'>E-mail: {antonio}@inf.ufg.br</p>

# Description
The repository contains all development related to the master's thesis entitled "Split Learning as an enabler of wireless networks for future generations". Being developed through the Academic Master's Course in Computer Science - In-person - Goiânia, by the Institute of Informatics (INF) of the Federal University of Goiás (UFG).

# Table of Contents
- [Article Summary](#getting-started)
	- [Abstract](#abstract)
	- [Baselines](#baselines)
	- [Proposed Synchronization Algorithm](#proposed-synchronization-algorithm)
	- [Results](#results)
- [Replicating the Experiment](#replicating-the-experiment)
	- [Requirements](#requirements)
	- [Preparing Environment](#preparing-environment)
 		- [Simulations](#simulations)
 	- [Run Experiments](#run-experiments)
  		- [Trains and Tests](#trains-and-tests)     
  		

- [How to cite](#how-to-cite)

## Abstract

Split Learning (SL) is a promising approach as an effective solution to data security and privacy concerns in training Deep Neural Networks (DNN), due to its approach characteristics of combining raw data security and the division of the model between client devices and central server.
Providing to minimize the risks of leaks and attacks, while keeping deep neural network training viable on devices with limited edge capabilities.
However, this split model allows for an increase in the communication flow between edge devices (distributed) and the server (aggregator), leaving an open question about communication overhead.
This dissertation covers the inference of the communication overload problem. Through a case study of offline integration with distributed learning of Split Learning by training a convolutional neural network (Convolutional Neural Network - CNN) and MNIST dataset. And the NS3 simulator, with characteristics of a Wi-Fi network environment with IoT device nodes and an Access Point.
In this integrated scenario, network experiments are simulated with distance variations of 10, 50 and 100 mt, powers of 10, 30 and 50 dBm and loss exponents of 2, 3 and 4 dB. Based on the network output results, with regard to latency, a policy was defined that values ​​above 4 seconds are considered timeouts and are not included in machine learning experiments. As well, training and testing was carried out on the split learning model, observing the impacts on accuracies and loss rates.

[Back to TOC](#table-of-contents)

## Baselines

The methods defined as baselines for our proposal: (1) the use of the asynchronous algorithm in training the Split Learning model and (2) the definition of training based on events provided from the simulation in NS3 based on the established latency policy. The training uses the file provided by the simulation (simulator_ns3.csv). 10 devices are simulated and the established policy for network latency is nodes below 04 seconds.

[Back to TOC](#table-of-contents)

## Proposed Synchronization Algorithm

The proposed synchronization algorithm was included in `clientX_async.py` clients in order to minimize the constant exchange of communication between clients and server in the Split Learning structure as a means to reduce communication overhead.

```python
class AsynchronousSplitLearning:
    def __init__(self, client_models, server_model, num_epoch, num_batch, K, lthred):
        self.state = 'A'
        self.client_models = client_models
        self.server_model = server_model
        self.num_epoch = num_epoch
        self.num_batch = num_batch
        self.K = K
        self.lthred = lthred
        self.total_loss = 0

    def split_forward(self, state, data, target, criterion):
        if state == 'C':
            act, y_star = None, None
        else:
            act = sum(client_model(data) for client_model in self.client_models) / len(self.client_models)
            y_star = target
        outputs = self.server_model(act)
        loss = criterion(outputs, target)
        return loss

    def split_backward(self, state, loss, optimizer):
        loss.backward()
        optimizer.step()

    def update_state(self, total_loss):
        last_update_loss = total_loss / (self.num_batch * self.K)
        delta_loss = last_update_loss - (total_loss / (self.num_batch * self.K))
        if delta_loss <= self.lthred:
            self.state = 'A'
        else:
            self.state = 'B' if self.state == 'A' else 'C'
        return self.state

    def train(self, train_loader, criterion, optimizer, latencies, delta_t, error_rate, device):
        for epoch in range(1, self.num_epoch + 1):
            total_loss = 0
            for client in range(1, self.K + 1):
                for batch_idx, (data, target) in enumerate(train_loader):
                    data, target = data.to(device), target.to(device)
                    optimizer.zero_grad()

                    latency, latency_val = introduce_latency(latencies, delta_t)
                    if latency is None:
                        continue

                    loss = self.split_forward(self.state, data, target, criterion)
                    total_loss += loss.item()
                    self.split_backward(self.state, loss, optimizer)
            self.state = self.update_state(total_loss)

```
[Back to TOC](#table-of-contents)

## Results
### Results in the communication network environment.

Demonstrations of results within the scope of simulation on the Wi-Fi network, graphically presenting: latencies, transfer rates, packet loss rates and energy consumption.

<p align='center'>
    <img src='/images/figure1.png' width='500'>
</p>    
<p align='center'>
    <figurecaption>
        Fig. 1. Latencys.
    </figurecaption>
</p>

Figure 1 shows the result of latencies in the network simulation.


<p align='center'>
    <img src='/images/figure2.png' width='500'>
</p>    
<p align='center'>
    <figurecaption>
        Fig. 2. Packet Losses.
    </figurecaption>
</p>

Figure 2 shows the result of packet losses in the network simulation.


<p align='center'>
    <img src='/images/figure3.png' width='500'>
</p>    
<p align='center'>
    <figurecaption>
        Fig. 3. Throughput.
    </figurecaption>
</p>

Figure 3 demonstrates the throughput results in the network simulation.

<p align='center'>
    <img src='/images/figure4.png' width='500'>
</p>    
<p align='center'>
    <figurecaption>
        Fig. 4. Throughput.
    </figurecaption>
</p>

Figure 4 demonstrates the energy consumption results in the network simulation.

### Results in the machine learning environment with training and testing

<p align='center'>
    <img src='/images/figure5.png' width='500'>
</p>    
<p align='center'>
    <figurecaption>
        Fig. 5. Accuracy per Round.
    </figurecaption>
</p>
Figure 5 shows the result of accuracy per round during training and testing in the Split Learning learning model.

<p align='center'>
    <img src='/images/figure6.png' width='500'>
</p>    
<p align='center'>
    <figurecaption>
        Fig. 6. Processing Accuracy.
    </figurecaption>
</p>

Figure 6 demonstrates the processing accuracy results during training and testing in the Split Learning learning model.

[Back to TOC](#table-of-contents)

# Replicating The Experiment

## Requirements

- GNU (>=8.0.0)
  
  command to know the version of KERNEL, GCC and GNU Binutils in the terminal
  ```bash
	cat /proc/version
  ```
- GCC (>=11.4.0)
  
  commands to know the GCC version in the terminal.
  ```bash
	gcc --version
	ls -l /usr/bin/gcc*
  ```
- CMAKE (>=3.24)
  
  command to know the version of CMAKE in the terminal.
  ```bash
	cmake --version
  ```
- python (>=3.11.5)
  
  commands to know the version of PYTHON in the terminal.
  ```bash
	python --version
	python3 --version
  ```
- [ns-allinone (3.42)](https://www.nsnam.org/releases/ns-3-42/download/ )
  
[Back to TOC](#table-of-contents)

## Preparing Environment

Start by cloning this repository into the NS3 `scracth` folder.

```bash
git clone https://github.com/cleyber-bezerra/SplitLearning-Async-NS3.git
```

The first step is to build the `ns-3.42` of NS3.

```bash
./ns3 configure --enable-examples
./ns3 build
```
[Back to TOC](#table-of-contents)

### Simulations

Then, compile the source code from the ns-3 `scratch` files.

```bash
./ns3 run scratch/SplitLearning-Async-NS3/my_wifi_ap_net_rand.cc
```
or inside the SplitLearning-NS3 folder run

```bash
./simulator_net.sh
```


> The name of the generated file will follow the pattern `simulator_ns3.csv`, and will be located in the internal path results/csv/ns3.

```bash
pip install numpy pandas toch tochvision matplotlib tqdm
```

We can then begin the process of training and testing the machine learning model.

[Back to TOC](#table-of-contents)

## Run Experiments
### Trains and Tests
To carry out training and testing in `synchronous` or `asynchronous` environments.

```bash
cd SplitLearning-Async-NS3
python run.py sync
```
or

```bash
cd SplitLearning-Async-NS3
python run.py async 
```
1) The generated CSV files will follow the following names `result_train_sync.csv` or `result_train_async.csv`, and will be located in the csv/ia folder.
2) The names of the generated graphic files will follow the pattern `net_*.png`, and will be located in the `images` folder.

Note: training and testing in synchronous/asynchronous environments is 1 round and with ID clients where the latency in the network simulation was less than 4 seconds.

[Back to TOC](#table-of-contents)

## How to cite
It

```
@INPROCEEDINGS{reis2024-xxx-xxx-xx,
    author={Reis, Cleyber B., Antonio O.},
    booktitle={{2024 )}},
    title={{D}},
    year={2024}, volume={}, number={}, pages={1-5},
    doi={10.}
}
```
[Back to TOC](#table-of-contents)

