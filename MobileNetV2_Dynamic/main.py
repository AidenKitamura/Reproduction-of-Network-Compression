# -*- coding=utf-8 -*-

import os
import torch
from tqdm import tqdm
from loguru import logger
from models import resnet34
from utils import *

import torch
import torchvision
import torchvision.transforms as transforms

from models.mobilenetv2 import MobileNetV2

def watch_nan(x: torch.Tensor):
    if torch.isnan(x).any():
        raise ValueError('found NaN: ' + str(x))


def temp_load_data():
    print('==> Preparing data..')
    transform_train = transforms.Compose([
        transforms.RandomCrop(32, padding=4),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])

    trainset = torchvision.datasets.CIFAR10(
        root='./data', train=True, download=True, transform=transform_train)
    trainloader = torch.utils.data.DataLoader(
        trainset, batch_size=150, shuffle=True, num_workers=0)

    testset = torchvision.datasets.CIFAR10(
        root='./data', train=False, download=True, transform=transform_test)
    testloader = torch.utils.data.DataLoader(
        testset, batch_size=128, shuffle=False, num_workers=0)
    
    return trainloader, testloader

if __name__ == '__main__':
    name = 'resnet34-imagenette2'
    os.makedirs('log', exist_ok=True)
    os.makedirs('ckpts', exist_ok=True)
    log_path = os.path.join('log', name + '.log')
    if os.path.isfile(log_path):
        os.remove(log_path)
    logger.add(log_path)
    # net = resnet34(num_classes=10).cuda()
    net = MobileNetV2().cuda()
    # train_dl = imagenette2('train')
    # valid_dl = imagenette2('val')
    train_dl, valid_dl = temp_load_data()
    for pruning_rate in [0.8, 0.6, 0.4]:
        logger.info('set pruning rate = %.1f' % pruning_rate)
        # set pruning rate
        for m in net.modules():
            if hasattr(m, 'rate'):
                m.rate = pruning_rate
        optimizer = torch.optim.SGD(net.parameters(), lr=0.1, momentum=0.9, weight_decay=0.0001)
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [30, 60, 80], 0.2)
        loss_function = torch.nn.CrossEntropyLoss()
        best_accuracy = 0
        for epoch in range(100):
            with tqdm(train_dl) as train_tqdm:
                train_tqdm.set_description_str('{:03d} train'.format(epoch))
                net.train()
                meter = AccuracyMeter(topk=(1, 5))
                for images, labels in train_tqdm:
                    images = images.cuda()
                    labels = labels.cuda()
                    output = net(images)
                    watch_nan(output)
                    meter.update(output, labels)
                    train_tqdm.set_postfix(meter.get())
                    optimizer.zero_grad()
                    loss1 = loss_function(output, labels)
                    loss2 = 0
                    for m in net.modules():
                        if hasattr(m, 'loss') and m.loss is not None:
                            loss2 += m.loss
                    loss = loss1 + 1e-8 * loss2
                    loss.backward()
                    optimizer.step()
                logger.info('{:03d} train result: {}'.format(epoch, meter.get()))
            with tqdm(valid_dl) as valid_tqdm:
                valid_tqdm.set_description_str('{:03d} valid'.format(epoch))
                net.eval()
                meter = AccuracyMeter(topk=(1, 5))
                with torch.no_grad():
                    for images, labels in valid_tqdm:
                        images = images.cuda()
                        labels = labels.cuda()
                        output = net(images)
                        watch_nan(output)
                        meter.update(output, labels)
                        valid_tqdm.set_postfix(meter.get())
                logger.info('{:03d} valid result: {}'.format(epoch, meter.get()))
            if (epoch + 1) % 10 == 0:
                torch.save(net.state_dict(), 'ckpts/%.1f-latest.pth' % pruning_rate)
                logger.info('saved to ckpts/%.1f-latest.pth' % pruning_rate)
            if best_accuracy < meter.top():
                best_accuracy = meter.top()
                torch.save(net.state_dict(), 'ckpts/%.1f-best.pth' % pruning_rate)
                logger.info('saved to ckpts/%.1f-best.pth' % pruning_rate)
            scheduler.step()