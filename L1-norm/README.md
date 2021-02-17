# Reproduction code for L1-norm pruning

A folder containing the code to run the pruning tool for Resnet18 on CIFAR10

# Torch-Pruning

A pytorch toolkit for structured neural network pruning and layer dependency maintaining

This tool will automatically detect and handle layer dependencies (channel consistency) during pruning. It is able to handle various network architectures such as DenseNet, ResNet, and Inception. See [examples/test_models.py](https://github.com/VainF/Torch-Pruning/blob/master/examples/test_models.py) for more supported models. 

# How to run the code

Execute run.py and enter the necessary arguments.

The following commands provide an example to train, test and prune the model:

### 1. Train the model
```bash
cd examples
python run.py --mode train # 11.1M, Acc=0.9248
```

### 2. Pruning and fintuning
```bash
python run.py --mode prune --round 1 --total_epochs 30 --step_size 20 # 4.5M, Acc=0.9229
python run.py --mode prune --round 2 --total_epochs 30 --step_size 20 # 1.9M, Acc=0.9207
python run.py --mode prune --round 3 --total_epochs 30 --step_size 20 # 0.8M, Acc=0.9176
python run.py --mode prune --round 4 --total_epochs 30 --step_size 20 # 0.4M, Acc=0.9102
python run.py --mode prune --round 5 --total_epochs 30 --step_size 20 # 0.2M, Acc=0.9011
...