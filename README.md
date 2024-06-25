# SplitLearning-Async-NS3

<p align='center' style="margin-bottom: -4px">Cleyber Bezerra dos Reis<sup>1</sup>, Antonio Oliveira-JR<sup>1</sup></p>
<p align='center' style="margin-bottom: -4px"><sup>1</sup>Instituto de Informática, Universidade Federal de Goiás</p>
<p align='center' style="margin-bottom: -4px">E-mail: {cleyber.bezerra}@discente.ufg.br</p>
<p align='center'>E-mail: {antonio}@inf.ufg.br</p>

# Table of Contents
- [Article Summary](#getting-started)
	- [Abstract](#abstract)
	- [Baselines](#baselines)
	- [Proposed placement algorithm (Async)](#proposed-placement-algorithm)
	- [Results](#results)
- [Replicating the Experiment](#replicating-the-experiment)
	- [Requirements](#requirements)
	- [Preparing Environment](#preparing-environment)
 		- [Simulations](#simulations)
 	- [Run Experiments](#run-experiments)
  		- [Trains and Tests](#trains-tests)
  		- [Optimization Model](#optimization-model)      
  		

- [How to cite](#how-to-cite)

## Abstract

Split Learning (SL) is a promising approach as an effective solution to data security and privacy concerns in training Deep Neural Networks (DNN), due to its approach characteristics of combining raw data security and the division of the model between client devices and central server.
Providing to minimize the risks of leaks and attacks, while keeping deep neural network training viable on devices with limited edge capabilities.
However, this split model allows for an increase in the communication flow between edge devices (distributed) and the server (aggregator), leaving an open question about communication overhead.
This dissertation covers the inference of the communication overload problem. Through a case study of offline integration with distributed learning of Split Learning by training a convolutional neural network (Convolutional Neural Network - CNN) and MNIST dataset. And the NS3 simulator, with characteristics of a Wi-Fi network environment with IoT device nodes and an Access Point.
In this integrated scenario, network experiments are simulated with distance variations of 10, 50 and 100 mt, powers of 10, 30 and 50 dBm and loss exponents of 2, 3 and 4 dB. Based on the network output results, with regard to latency, a policy was defined that values ​​above 4 seconds are considered timeouts and are not included in machine learning experiments. As well, training and testing was carried out on the split learning model, observing the impacts on accuracies and loss rates.


## Baselines
Two

## Results
The results demonstrate the presentation of latencies, transfer rates, packet loss rates and energy consumption...

### {Eq}
EQ  area $\mathcal{A}_{(MxN)}$ and the lower bound QoS  to be reached. The algorithm iterates by incrementing the number of UAVs to find how many satisfy the $QoS_{bound}$.
### {Density-oriented UAVs placement algorithm (DO)}
DO  of EDs $\beta_{(MxN)}$. Initially, the area density information is calculated, then $\beta_{(MxN)}$ is linearized, sorted in descending order, and assigned to $\rho_{(L)}$. The algorithm iterates until it reaches the target bound $QoS_{bound}$. In each iteration, a new UAV is added and placed in the corresponding position $\rho{(l)}$ of a matrix of $\alpha_{(m,n)}$, where $l$ represents the linear index that corresponds to a $(m,n)$. The DO outputs a UAV placement map $\alpha_{(m,n)}$.

## Proposed placement algorithm (Async)

<img src="/images/OP_Algorithm.png" width="500">
Figure 1:L...

The OP algorithm presents

[Back to TOC](#table-of-contents)

# Replicating The Experiment

## Requirements

- GNU (>=8.0.0)
- CMAKE (>=3.24)
- python (3.11.4)
- ns-allinone (3.42)
  
[Back to TOC](#table-of-contents)

## Preparing Environment

Start by cloning this repository into the NS3 `scracth` folder.

```bash
git clone https://github.com/cleyber-bezerra/SplitLearning-Async-NS3.git
```

The first step is to build the ns-3.42 of NS3.

```bash
./ns3 configure --enable-examples
./ns3 build
```
### Simulations

Then, compile the source code from the ns-3 `scratch` files.

```bash
./ns3 run scratch/SL/my_wifi_ap_net_rand.cc {{{ --nDevices=30 --seed=1 --nGateways=4 }}}
```

> The name of the generated file will follow the pattern `simulator_ns3.csv`, and will be located in the internal path results/csv/ns3.

```bash
pip install numpy pandas toch tochvision matplotlib 
```

We can then begin the process of training and testing the machine learning model.

## Run Experiments
### Trains and Tests

a. Gene.

```bash
cd SplitLearning-Async-NS3
python server_overhead_mnist.py {ASYNC=0}
```
> The name of the generated file will follow the pattern `result_train_sync.csv`, and will be located in the internal path results/csv/ia.
> The names of the generated files will follow the pattern `net_*.png`, and will be located in the internal path results/img.

b. Gene.

```bash
python server_overhead_mnist.py {ASYNC=1}
```
> The name of the generated file will follow the pattern `result_train_async.csv`, and will be located in the internal path results/csv/ia.
> The names of the generated files will follow the pattern `net_*.png`, and will be located in the internal path results/img.

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

