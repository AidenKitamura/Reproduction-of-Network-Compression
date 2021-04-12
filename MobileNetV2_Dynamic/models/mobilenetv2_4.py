'''MobileNetV2 in PyTorch.

See the paper "Inverted Residuals and Linear Bottlenecks:
Mobile Networks for Classification, Detection and Segmentation" for more details.
'''
import torch
from torch.functional import norm
import torch.nn as nn
import torch.nn.functional as F

from . import ChannelPruning

def conv3x3(in_planes: int,
            out_planes: int,
            stride: int = 1,
            groups: int = 1,
            dilation: int = 1,
            normalization: str = '',
            activation: str = '') -> nn.Module:
    layers = []
    layers.append(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=3,
                  stride=stride,
                  groups=groups,
                  padding=dilation,
                  dilation=dilation,
                  bias=False)
    )
    if normalization == 'bn':
        layers.append(nn.BatchNorm2d(out_planes))
    if activation == 'relu':
        layers.append(nn.ReLU())
    return ChannelPruning(*layers)


def conv1x1(in_planes: int,
            out_planes: int,
            stride: int = 1,
            normalization: str = '',
            activation: str = '') -> nn.Module:
    layers = []
    layers.append(
        nn.Conv2d(in_planes,
                  out_planes,
                  kernel_size=1,
                  stride=stride,
                  bias=False)
    )
    if normalization == 'bn':
        layers.append(nn.BatchNorm2d(out_planes))
    if activation == 'relu':
        layers.append(nn.ReLU())
    return ChannelPruning(*layers)


class Block(nn.Module):
    '''expand + depthwise + pointwise'''
    def __init__(self, in_planes, out_planes, expansion, stride):
        super(Block, self).__init__()
        self.stride = stride

        planes = expansion * in_planes
        # self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn1 = nn.BatchNorm2d(planes)
        # self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, groups=planes, bias=False)
        # self.bn2 = nn.BatchNorm2d(planes)
        # self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn3 = nn.BatchNorm2d(out_planes)

        self.conv1 = conv1x1(in_planes, planes, normalization='bn')
        self.conv2 = conv3x3(planes, planes, normalization='bn', groups=planes,stride=stride)
        # self.conv3 = conv1x1(planes, out_planes, normalization='bn')  # dont prune in this layer
        # self.conv3 = conv1x1(planes, out_planes, normalization='bn')
        
        self.conv3 = nn.Conv2d(planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False)
        self.bn3 = nn.BatchNorm2d(out_planes)


        self.shortcut = nn.Sequential()
        if stride == 1 and in_planes != out_planes:
            # self.shortcut = conv1x1(in_planes, out_planes, normalization='bn')
            
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=1, padding=0, bias=False),
                nn.BatchNorm2d(out_planes),
            )

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = F.relu(self.bn2(self.conv2(out)))
        # out = self.bn3(self.conv3(out))
        # out = out + self.shortcut(x) if self.stride==1 else out

        out = F.relu(self.conv1(x))
        out = F.relu(self.conv2(out))
        out = self.conv3(out)
        out = self.bn3(out)
        out = out + self.shortcut(x) if self.stride==1 else out

        return out


class MobileNetV2(nn.Module):
    # (expansion, out_planes, num_blocks, stride)
    cfg = [(1,  16, 1, 1),
           (6,  24, 2, 1),  # NOTE: change stride 2 -> 1 for CIFAR10
           (6,  32, 3, 2),
           (6,  64, 4, 2),
           (6,  96, 3, 1),
           (6, 160, 3, 2),
           (6, 320, 1, 1)]

    def __init__(self, num_classes=10):
        super(MobileNetV2, self).__init__()
        # NOTE: change conv1 stride 2 -> 1 for CIFAR10
        # self.conv1 = nn.Conv2d(3, 32, kernel_size=3, stride=1, padding=1, bias=False)
        # self.bn1 = nn.BatchNorm2d(32)
        # self.layers = self._make_layers(in_planes=32)
        # self.conv2 = nn.Conv2d(320, 1280, kernel_size=1, stride=1, padding=0, bias=False)
        # self.bn2 = nn.BatchNorm2d(1280)

        self.conv1 = conv3x3(3,32, normalization='bn')
        self.layers = self._make_layers(in_planes=32)
        self.conv2 = conv1x1(320, 1280, normalization='bn')

        self.linear = nn.Linear(1280, num_classes)

    def _make_layers(self, in_planes):
        layers = []
        for expansion, out_planes, num_blocks, stride in self.cfg:
            strides = [stride] + [1]*(num_blocks-1)
            for stride in strides:
                layers.append(Block(in_planes, out_planes, expansion, stride))
                in_planes = out_planes
        return nn.Sequential(*layers)

    def forward(self, x):
        # out = F.relu(self.bn1(self.conv1(x)))
        # out = self.layers(out)
        # out = F.relu(self.bn2(self.conv2(out)))
        # print(x.shape)
        out = F.relu(self.conv1(x))
        # print(out.shape)
        out = self.layers(out)
        # print(out.shape)
        out = F.relu(self.conv2(out))
        # print(out.shape)
        # NOTE: change pooling kernel_size 7 -> 4 for CIFAR10
        out = F.avg_pool2d(out, 4)
        out = out.view(out.size(0), -1)
        # print(out.shape)
        out = self.linear(out)
        return out


def test():
    net = MobileNetV2()
    x = torch.randn(2,3,32,32)
    y = net(x)
    print(y.size())

# test()
