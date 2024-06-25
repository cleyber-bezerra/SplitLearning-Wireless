# SplitLearning-Async-NS3

<p align='center' style="margin-bottom: -4px">Cleyber Bezerra dos Reis<sup>1</sup>, Antonio Oliveira-JR<sup>1</sup></p>
<p align='center' style="margin-bottom: -4px"><sup>1</sup>Instituto de Informática, Universidade Federal de Goiás</p>
<p align='center' style="margin-bottom: -4px">E-mail: {cleyber.bezerra}@discente.ufg.br</p>
<p align='center'>E-mail: {antonio}@inf.ufg.br</p>

# Table of Contents
- [Article Summary](#getting-started)
	- [Abstract](#abstract)
	- [Baselines](#baselines)
	- [Proposed placement algorithm](#proposed-placement-algorithm)
	- [Results](#results)
- [Replicating the Experiment](#replicating-the-experiment)
	- [Requirements](#requirements)
	- [Preparing Environment](#preparing-environment)
 		- [Simulations](#simulations)
 	- [Run Experiments](#run-experiments)
  		- [Generating Input Data](#generating-input-data)
  		- [Optimization Model](#optimization-model)      
  		

- [How to cite](#how-to-cite)

## Abstract

Split Learning (SL) is a promising approach as an effective solution to data security and privacy concerns in training Deep Neural Networks (DNN), due to its approach characteristics of combining raw data security and the division of the model between client devices and central server.
Providing to minimize the risks of leaks and attacks, while keeping deep neural network training viable on devices with limited edge capabilities.
However, this split model allows for an increase in the communication flow between edge devices (distributed) and the server (aggregator), leaving an open question about communication overhead.
This dissertation covers the inference of the communication overload problem. Through a case study of offline integration with distributed learning of Split Learning by training a convolutional neural network (Convolutional Neural Network - CNN) and MNIST dataset. And the NS3 simulator, with characteristics of a Wi-Fi network environment with IoT device nodes and an Access Point.
In this integrated scenario, network experiments are simulated with distance variations of 10, 50 and 100 mt, powers of 10, 30 and 50 dBm and loss exponents of 2, 3 and 4 dB. Based on the network output results, with regard to latency, a policy was defined that values ​​above 4 seconds are considered timeouts and are not included in machine learning experiments. As well, training and testing was carried out on the split learning model, observing the impacts on accuracies and loss rates.
The results demonstrate the presentation of latencies, transfer rates, packet loss rates and energy consumption...

## Baselines
Two

### {Equidistant UAVs placement algorithm (EQ)}
EQ  area $\mathcal{A}_{(MxN)}$ and the lower bound QoS  to be reached. The algorithm iterates by incrementing the number of UAVs to find how many satisfy the $QoS_{bound}$.
### {Density-oriented UAVs placement algorithm (DO)}
DO  of EDs $\beta_{(MxN)}$. Initially, the area density information is calculated, then $\beta_{(MxN)}$ is linearized, sorted in descending order, and assigned to $\rho_{(L)}$. The algorithm iterates until it reaches the target bound $QoS_{bound}$. In each iteration, a new UAV is added and placed in the corresponding position $\rho{(l)}$ of a matrix of $\alpha_{(m,n)}$, where $l$ represents the linear index that corresponds to a $(m,n)$. The DO outputs a UAV placement map $\alpha_{(m,n)}$.

## Proposed placement algorithm

<img src="/images/OP_Algorithm.png" width="500">

The OP algorithm presents our approach for positioning UAVs-based gateways following the optimization model in section 2. The algorithm receives as input the area for positioning devices ($\mathcal{A}_{(MxN)}$), the UAV placement area ($\mathcal{V}_{(XxYxZ)}$), and the expected QoS limit ($\rho^{QoS}_l$). In line 1, the devices are distributed following a realistic spatial distribution (here).tions set $\mathcal{P}$, the SF (spreading factor) and TP (transmission power) configurations set $\mathcal{C}$, and the slices associations set $\mathcal{S}$. Finally, the resulting data are modeled in the NS-3, and the simulation is performed.
[Back to TOC](#table-of-contents)

## Results
The 

[Back to TOC](#table-of-contents)

# Replicating The Experiment

## Requirements

- GNU (>=8.0.0)
- CMAKE (>=3.24)
- python (3.11.4)
- ns-allinone-3.42
  
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

The following Python packages are needed to execute the experiment.

```bash
pip install numpy pandas toch tochvision matplotlib 
```

We can then begin the process of training and testing the machine learning model.

## Run Experiments (trains)
### Generating Input Data

a. Gene.

```bash
cd SplitLearning-Async-NS3
python server_overhead_mnist.py
```
> The generated file names will follow the pattern `equidistantPlacement_xx.dat`, where `xx` is the number of virtual positions for UAV deployment.

b. Gene.

```bash
python server_open_filter_mnist.py
```
> The generated file names will follow the pattern `endDevices_LNM_Placement_1s+30d.dat`, where `1s` and `30d` follow the adopted parameters for seed and devices.

c. Gene.
```bash
./server_open_filter_mnist.py 
```
> The

[Back to TOC](#table-of-contents)



This  

```bash
./ns-3/build/scratch/ns3.36-eq-experiment-debug --nDevices=30 --seed=1 --nGateways=25
./ns-3/build/scratch/ns3.36-do-experiment-debug --nDevices=30 --seed=1 --nGateways=25
./ns-3/build/scratch/ns3.36-op-experiment-debug --nDevices=30 --seed=1 --nGateways=25
```

The simulation output can be found in the directory [./data/results/](./data/results/). 
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

