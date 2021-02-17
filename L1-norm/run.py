import sys
sys.path.insert(0,'../utils/')
from data_loader_l1norm import get_dataloader
from evaluate_l1norm import eval
sys.path.insert(0,'./resnet18_pruning_cifar10.py')
import torch
from resnet_model import ResNet18
import numpy as np
from resnet18_pruning_cifar10 import L1_norm
import argparse



parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--rounds', type=int, default=1)

args = parser.parse_args()

l1 = L1_norm(args.step_size,args.total_epochs,args.verbose,args.rounds) #step_size,total_epochs,verbose,rounds

def main():
    '''Takes in the user input and initializes the L1_norm pruning class.

    Takes in the user input, initializes the class L1_norm and execute the corresponding mode on model.
    '''
    train_loader, test_loader = get_dataloader()
    if args.mode=='train':
        l1.round=0
        model = ResNet18(num_classes=10)
        l1.train_model(model, train_loader, test_loader)
    elif args.mode=='prune':
        previous_ckpt = 'resnet18-round%d.pth'%(l1.round-1)
        print("Pruning round %d, load model from %s"%( l1.round, previous_ckpt ))
        model = torch.load( previous_ckpt )
        l1.prune_model(model)
        print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        l1.train_model(model, train_loader, test_loader)
    elif args.mode=='test':
        ckpt = 'resnet18-round%d.pth'%(l1.round)
        print("Load model from %s"%( ckpt ))
        model = torch.load( ckpt )
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n"%(acc))

if __name__=='__main__':
    main()
