# Test-time Training
Replication of Code for paper [Test-Time Training with Self-Supervision for Generalization under Distribution Shifts](https://yueatsprograms.github.io/ttt/home.html).

This code is meant to do more experiments and play around with various components of the original paper.

### Setup and Installation

Please follow instructions [here](https://github.com/bpiyush/test-time-training/blob/main/setup/instructions.md) to setup dependencies.

### Datasets

This repository currently only has support to setup CIFAR-10 dataset (and variants). More will follow soon. For required CIFAR datasets, please follow the instructions:

Setup a common folder for all datasets, for example,
```bash
DATASET_DIR=/scratch/users/piyushb/test-time-training/datasets
mkdir -p $DATASET_DIR
```

#### CIFAR-10-C
To download and setup this, modify `datasets/download/cifar-10-c.sh` and add your own `DATASET_DIR`. Then, run
```bash
bash datasets/download/cifar-10-c.sh
```

#### CIFAR-10.1
To download and setup this, modify `datasets/download/cifar-10.1.sh` and add your own `DATASET_DIR`. Then, run
```bash
bash datasets/download/cifar-10.1.sh
```

