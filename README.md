# Reproduction-of-Network-Compression
This repository includes reproductions for Mobile Net V2, Dynamic Channel Pruning, L1-Norm based Pruning, and Geometric Median Pruning. When doing pruning, all implementations are done according to our report.
# Environment Configuration
Pytorch 1.4.0
Python 3.6.10
# Usage
### Dynamic Channel Pruning
We assume you already switched to MobileNetV2_Dynamic folder
```
python main_cfg.py --rate 0.02 --strategy linear
```
where rate can be specified and strategy can be linear, exp or constant
### L1-Norm based Pruning
We assume you already switched to MobileV2_L1/examples folder
```
python prune_mobilev2_cifar10.py --mode prune --round 1 --total_epochs 30 --step_size 20 --prune_rate 0.02 --prune_method exp 
```
where rate can be specified and prune_method can be linear, exp or constant
### Geometric Median Pruning
We assume you already switched to geometric_median folder
```
python pruning_cifar10.py  ./input/cifar.python --dataset cifar10 --save_path ./output --schedule  60 120 160 --gammas 0.2 0.2 0.2 --lr 0.001 --decay 0.0005 --batch_size 128 --rate_norm 1 --layer_begin 0  --layer_end 54 --layer_inter 3 --epoch_prune 1 --epochs 100 --rate_dist 0.02 --prune_method constant
```
where rate_dist (pruning rate) can be specified and prune_method can be linear, exp or constant