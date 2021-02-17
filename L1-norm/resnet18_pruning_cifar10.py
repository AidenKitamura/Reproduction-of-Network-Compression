import sys, os
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
sys.path.insert(0,'../utils/')
from data_loader_l1norm import get_dataloader
from evaluate_l1norm import eval
from resnet_model import ResNet18
import resnet_model as resnet
import torch_pruning as tp
import argparse
import torch
from torchvision.datasets import CIFAR10
from torchvision import transforms
import torch.nn.functional as F
import torch.nn as nn 
import numpy as np 



class L1_norm(object):
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
    def __init__(self,step_size,total_epochs,verbose,rounds):
        super(L1_norm, self).__init__()
        self.step_size = step_size
        self.total_epochs = total_epochs
        self.verbose = verbose
        self.rounds = rounds

    def train_model(self, model, train_loader, test_loader):
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
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, self.step_size, 0.1)
        model.to(device)

        best_acc = -1
        for epoch in range(self.total_epochs):
            model.train()
            print(len(train_loader))
            for i, (img, target) in enumerate(train_loader):
                print(i)
                img, target = img.to(device), target.to(device)
                optimizer.zero_grad()
                out = model(img)
                loss = F.cross_entropy(out, target)
                loss.backward()
                optimizer.step()
                if i%10==0 and self.verbose:
                    print("Epoch %d/%d, iter %d/%d, loss=%.4f"%(epoch, self.total_epochs, i, len(train_loader), loss.item()))
            model.eval()
            acc = eval(model, test_loader)
            print("Epoch %d/%d, Acc=%.4f"%(epoch, self.total_epochs, acc))
            if best_acc<acc:
                torch.save( model, 'resnet18-round%d.pth'%(self.rounds) )
                best_acc=acc
            scheduler.step()
        print("Best Acc=%.4f"%(best_acc))

    def prune_model(self,model):
        '''Prune the model and saves it.

        Prunes the model using Torch_pruning toolkit.
        Produces the accuracy after pruning and the number of pruned parameters.

        Args:
            model: a instance of class ResNet18 whose parameters has been trained
        '''
        model.cpu()
        DG = tp.DependencyGraph().build_dependency( model, torch.randn(1, 3, 32, 32) )
        def prune_conv(conv, amount=0.2):
            #weight = conv.weight.detach().cpu().numpy()
            #out_channels = weight.shape[0]
            #L1_norm = np.sum( np.abs(weight), axis=(1,2,3))
            #num_pruned = int(out_channels * pruned_prob)
            #pruning_index = np.argsort(L1_norm)[:num_pruned].tolist() # remove filters with small L1-Norm
            strategy = tp.strategy.L1Strategy()
            pruning_index = strategy(conv.weight, amount=amount)
            plan = DG.get_pruning_plan(conv, tp.prune_conv, pruning_index)
            plan.exec()
        
        block_prune_probs = [0.1, 0.1, 0.2, 0.2, 0.2, 0.2, 0.3, 0.3]
        blk_id = 0
        for m in model.modules():
            if isinstance( m, resnet.BasicBlock ):
                prune_conv( m.conv1, block_prune_probs[blk_id] )
                prune_conv( m.conv2, block_prune_probs[blk_id] )
                blk_id+=1
        return model    