import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))

from cifar_resnet import ResNet18
import cifar_resnet as resnet

import time
from libs.vis_utils import visualize_grid
import matplotlib.pyplot as plt
import torch_pruning as tp
import argparse
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 
import mobilenetv2

parser = argparse.ArgumentParser()
parser.add_argument('--mode', type=str, required=True, choices=['train', 'prune', 'test'])
parser.add_argument('--batch_size', type=int, default=256)
parser.add_argument('--verbose', action='store_true', default=False)
parser.add_argument('--total_epochs', type=int, default=100)
parser.add_argument('--step_size', type=int, default=70)
parser.add_argument('--round', type=int, default=1)

args = parser.parse_args()

"""An example of pruning Resnet-18 on CIFAR10.

    Provides methods to train, test and prune the model. 
    Produces accuracy results and the amount of pruned parameters at the end of execution.

    Attributes:
        mode: a string indicating the mode as 'train', 'test' and 'prune'
        batch_size: an int indicating the size of each batch
        total_epochs: an int indicating the number of training epochs
        step_size: an int that provides the step size for scheduler
        round: an int that provides the index of round during pruing process
"""

def get_dataloader():
    train_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=True, transform=transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ]), download=True),batch_size=args.batch_size, num_workers=2)
    test_loader = torch.utils.data.DataLoader(
        CIFAR10('./data', train=False, transform=transforms.Compose([
            transforms.ToTensor(),
        ]),download=True),batch_size=args.batch_size, num_workers=2)
    return train_loader, test_loader

def eval(model, test_loader):
    '''Evaluates the model and produces accuracy results.

    Args:
        model: a instance of class ResNet18
        test_loader: A dataloader that provides the whole test dataset
            in terms of torch tensor and grouped batch by batch

    Returns:
        A float that provides the accuracy of the model on CIFAR-10
    '''
    correct = 0
    total = 0
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    with torch.no_grad():
        for i, (img, target) in enumerate(test_loader):
            img = img.to(device)
            out = model(img)
            pred = out.max(1)[1].detach().cpu().numpy()
            target = target.cpu().numpy()
            correct += (pred==target).sum()
            total += len(target)
    return correct / total

def train_model(model, train_loader, test_loader):
    '''Train and test the model.

    Trains the model and saves the model after training epochs.

    Args:
        model: a instance of class ResNet18 whose parameters will be trained
        train_loader:A dataloader that provides the whole train dataset
            in terms of torch tensor and grouped batch by batch
        test_loader: A dataloader that provides the whole test dataset
            in terms of torch tensor and grouped batch by batch
    '''
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    optimizer = torch.optim.SGD(model.parameters(), lr=0.1, momentum=0.9, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, args.step_size, 0.1)
    model.to(device)

    best_acc = -1
    print('--train start')
    for epoch in range(args.total_epochs):
        print('--epoch %d'%(epoch))
        model.train()
        print(len(train_loader))
        for i, (img, target) in enumerate(train_loader):
            print('--enumerate %d'%i)
            img, target = img.to(device), target.to(device)
            optimizer.zero_grad()
            out = model(img)
            loss = F.cross_entropy(out, target)
            loss.backward()
            optimizer.step()
            if i%10==0 and args.verbose:
                print("Epoch %d/%d, iter %d/%d, loss=%.4f"%(epoch, args.total_epochs, i, len(train_loader), loss.item()))
        model.eval()
        acc = eval(model, test_loader)
        print("Epoch %d/%d, Acc=%.4f"%(epoch, args.total_epochs, acc))
        if best_acc<acc:
            torch.save( model, 'resnet18-round%d.pth'%(args.round) )
            best_acc=acc
        scheduler.step()
    print("Best Acc=%.4f"%(best_acc))

def prune_model(model):
    '''Prune the model.

    Prunes the model using Torch_pruning toolkit.
    Produces the accuracy after pruning and the number of pruned parameters.

    Args:
        model: a instance of class ResNet18 whose parameters has been trained
    '''
    DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 32, 32) )
    
    def plot_filters_single_channel(t, index):
        
        #kernels depth * number of kernels
        nplots = t.shape[0]*t.shape[1]
        ncols = 12
        
        nrows = 1 + nplots//ncols
        #convert tensor to numpy image
        npimg = np.array(t.numpy(), np.float32)
        
        count = 0
        fig = plt.figure(figsize=(ncols, nrows))
        
        #looping through all the kernels in each channel
        for i in range(t.shape[0]):
            if i not in index:
              continue
            for j in range(t.shape[1]):
                count += 1
                ax1 = fig.add_subplot(nrows, ncols, count)
                npimg = np.array(t[i, j].numpy(), np.float32)
                npimg = (npimg - np.mean(npimg)) / np.std(npimg)
                npimg = np.minimum(1, np.maximum(0, (npimg + 0.5)))
                ax1.imshow(npimg)
                ax1.set_title(str(i) + ',' + str(j))
                ax1.axis('off')
                ax1.set_xticklabels([])
                ax1.set_yticklabels([])
      
        plt.tight_layout()
        plt.show()

    def plot_weights(layer, index, single_channel = True, collated = False):
        
        #checking whether the layer is convolution layer or not 
        if isinstance(layer, nn.Conv2d):
          #getting the weight tensor data
          weight_tensor = layer.weight.data
          if single_channel:
              plot_filters_single_channel(weight_tensor, index)
        else:
          print("Can only visualize layers which are convolutional")
    
    def prune_conv(conv, amount=0.2):
        #weight = conv.weight.detach().cpu().numpy()
        #out_channels = weight.shape[0]
        #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
        #num_pruned = int(out_channels * pruned_prob)
        #pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
        strategy = tp.strategy.L1Strategy()
        pruning_index = strategy(conv.weight, amount=amount)
        
        # print('pruning index:', pruning_index)
        #visualize weights for alexnet - first conv layer
        # plot_weights(conv, pruning_index, single_channel = True)      
        
        plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
        # print('******pruning plan*******')
        # print(plan)
        plan.exec()
    
    block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
    blk_id = 0
    counter = 0
    for m in model.modules():
        if isinstance( m, mobilenetv2.Block ):
            counter+= 1
            # print("--------------Block %d-----------------"%counter)
            # print(m)
            if blk_id>len(block_prune_probs)-1:
              blk_id = 0
            else:
              prune_conv( m.conv1, block_prune_probs[blk_id])
              prune_conv( m.conv2, block_prune_probs[blk_id])
              blk_id+=1
    print("Num of blocks: %d"%counter)
    return model    

def main():
    '''Takes in the user input and initializes the L1_norm pruning class.

    Takes in the user input, initializes the class L1_norm and execute the corresponding mode on model.
    '''
    train_loader, test_loader = get_dataloader()
    if args.mode=='train':
        args.round=0
        model = ResNet18(num_classes=10)
        train_model(model, train_loader, test_loader)
    elif args.mode=='prune':
        # previous_ckpt = 'resnet18-round%d.pth'%(args.round-1)
        print("Pruning round %d, load model from %s"%( args.round, 'ckpt.pth' ))
        model = mobilenetv2.MobileNetV2()
        model_loaded = torch.load('ckpt.pth')
        # print(model_loaded.keys())
        # print(model_loaded['net'].keys())
        mobile = model_loaded['net'].copy()
        model.load_state_dict({k.replace('module.',''):v for k,v in mobile.items()})
        # print(model)
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters Before Pruning: %.1fM"%(params/1e6))
        print("**********************START PRUNING******************")
        # print(type(model))
        prune_model(model)
        print("**********************START TRAINING*****************")
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters After Pruning: %.1fM"%(params/1e6))
        train_model(model, train_loader, test_loader)
        torch.save( model, 'mobilev2_l1-round%d.pth'%(args.round) )
    elif args.mode=='test':
        start = time.time()
        ckpt = 'mobilev2_l1-round%d.pth'%(args.round)
        print("Load model from %s"%( ckpt ))
        model = torch.load( ckpt )
        params = sum([np.prod(p.size()) for p in model.parameters()])
        print("Number of Parameters: %.1fM"%(params/1e6))
        acc = eval(model, test_loader)
        print("Acc=%.4f\n"%(acc))
        print(f"Run time: {(time.time() - start):.3f} s")

if __name__=='__main__':
    main()

