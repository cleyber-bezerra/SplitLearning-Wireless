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
 	- [Run Experiments](#run-experiments)
  		- [Generating Input Data](#generating-input-data)
  		- [Optimization Model](#optimization-model)      
  		- [Simulations](#simulations)

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

### UAVs placement

<p align='center'>
    <img src='/images/UAVs_Placements.png' width='500'>
</p>    
<p align='center'>
    <figurecaption>
        Fig. 1. UAV placement.
    </figurecaption>
</p>

Fig. 1 shows the area for UAV placement in a scenario with 60 EDs. Colored symbols represent UAVs for each placement method, and black circles represent devices.

### Number of UAV needed to serve the devices, ensuring the QoS level. 


### QoS
<p align='center'> <img src="/images/Heatmap_QoS.png" width="700"></p>
<p align='center'>
    <figurecaption>
        Fig. 3. QoS.
    </figurecaption>
</p>

Figure 3
[Back to TOC](#table-of-contents)

# Replicating The Experiment

## Requirements

- GNU (>=8.0.0)
- CMAKE (>=3.24)
- python (3.11.4)
- [SCIP Optimization Suite (8.0.3)](https://scipopt.org/index.php#download)
  
[Back to TOC](#table-of-contents)

## Preparing Environment

Start by cloning this repository.

```bash
git clone https://github.com/LABORA-INF-UFG/non3GPP_IoT_simulations.git iot-sim
cd iot-sim
```

The first step is to build the version 3.36 of NS3.

```bash
git clone https://github.com/nsnam/ns-3-dev-git ns-3
cd ns-3
git checkout ns-3.36

cp -r ../contrib/* ./contrib
cp -r ../scratch/* ./scratch

./ns3 configure --enable-examples
./ns3 build
```
Then, compile the source code from the ns-3 scratch files. The error messages presented at this step occur because we have not yet sent the appropriate execution parameters; ignore them.

```
./ns3 run scratch/ed-do-placement.cc
./ns3 run scratch/gw-do-placement.cc
./ns3 run scratch/op-prepare.cc
./ns3 run scratch/do-experiment.cc
./ns3 run scratch/eq-experiment.cc
./ns3 run scratch/op-experiment.cc
```

The following Python packages are needed to execute the experiment.

```bash
pip install pyomo pandas matplotlib blob
```

We can then start the experimentation process; after every step, you can check the generated files inside [data/](data/) folder.

## Run Experiments
### Generating Input Data

a. Generate the files with the virtual positions for UAV placement. You must configure the `verbose` parameter.

```bash
cd iot-sim
python eq-placement.py 1
```
> The generated file names will follow the pattern `equidistantPlacement_xx.dat`, where `xx` is the number of virtual positions for UAV deployment.

b. Generate the files with LoRa-ED positions using the NS3 script; you can modify the number of devices with the option `--nDevices=x` and the seed for the pseudo-random distribution of the devices with the option `--seed=y`.

```bash
./ns-3/build/scratch/ns3.36-ed-do-placement-debug --nDevices=30 --seed=1
```
> The generated file names will follow the pattern `endDevices_LNM_Placement_1s+30d.dat`, where `1s` and `30d` follow the adopted parameters for seed and devices.

c. Generate the files with UAV positions to the baseline Density-Oriented UAVs experiment.
```bash
./ns-3/build/scratch/ns3.36-gw-do-placement-debug --nDevices=30 --seed=1 --nGateways=4
```
> The generated file names will follow the pattern `densityOrientedPlacement_1s+30d+4g.dat`, where `1s`, `30d` and `4d` follow the adopted parameters for seed, devices and gateways(UAVs).

d. To finalize this step, generate the files with slice association of the devices and other optimization model input parameters.

```bash
./ns-3/build/scratch/ns3.36-op-prepare-debug --nDevices=30 --nGateways=25 --seed=1 --nPlanes=1
```
> The files containing the input settings for the optimization model are generated in the [./data/model/](./data/model/) folder.
[Back to TOC](#table-of-contents)

### Optimization Model

To . 

```bash
cd iot-sim
python model.py 25 1 30 1 0.9
```
[Back to TOC](#table-of-contents)

### Simulations

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

