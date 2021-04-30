from __future__ import division

import os, sys, shutil, time, random
import argparse
import torch
import torch.backends.cudnn as cudnn
import torchvision.datasets as dset
import torchvision.transforms as transforms
from utils import AverageMeter, RecorderMeter, time_string, convert_secs2time, timing
import mobilenetv2
import numpy as np
import pickle
from scipy.spatial import distance
import pdb
import time


parser = argparse.ArgumentParser(description='Trains ResNeXt on CIFAR or ImageNet',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('data_path', type=str, help='Path to dataset')
parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'imagenet', 'svhn', 'stl10'],
                    help='Choose between Cifar10/100 and ImageNet.')
# Optimization options
parser.add_argument('--epochs', type=int, default=300, help='Number of epochs to train.')
parser.add_argument('--batch_size', type=int, default=128, help='Batch size.')
parser.add_argument('--lr', type=float, default=0.1, help='The Learning Rate.')
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', type=float, default=0.0005, help='Weight decay (L2 penalty).')
parser.add_argument('--schedule', type=int, nargs='+', default=[150, 225],
                    help='Decrease learning rate at these epochs.')
parser.add_argument('--gammas', type=float, nargs='+', default=[0.1, 0.1],
                    help='LR is multiplied by gamma on schedule, number of gammas should be equal to schedule')
# Checkpoints
parser.add_argument('--print_freq', default=200, type=int, metavar='N', help='print frequency (default: 200)')
parser.add_argument('--save_path', type=str, default='./', help='Folder to save checkpoints and log.')
parser.add_argument('--resume', default='', type=str, metavar='PATH', help='path to latest checkpoint (default: none)')
parser.add_argument('--start_epoch', default=0, type=int, metavar='N', help='manual epoch number (useful on restarts)')
parser.add_argument('--evaluate', dest='evaluate', action='store_true', help='evaluate model on validation set')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--workers', type=int, default=2, help='number of data loading workers (default: 2)')
# random seed
parser.add_argument('--manualSeed', type=int, help='manual seed')
# compress rate
parser.add_argument('--rate_norm', type=float, default=0.9, help='the remaining ratio of pruning based on Norm')
parser.add_argument('--rate_dist', type=float, required=True, default=0.1, help='the reducing ratio of pruning based on Distance')
parser.add_argument('--layer_begin', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_end', type=int, default=1, help='compress layer of model')
parser.add_argument('--layer_inter', type=int, default=1, help='compress layer of model')
parser.add_argument('--epoch_prune', type=int, default=1, help='compress layer of model')
parser.add_argument('--use_state_dict', dest='use_state_dict', action='store_true', help='use state dcit or not')
parser.add_argument('--use_pretrain', dest='use_pretrain', action='store_true', help='use pre-trained model or not')
parser.add_argument('--pretrain_path', default='', type=str, help='..path of pre-trained model')
parser.add_argument('--dist_type', default='l2', type=str, choices=['l2', 'l1', 'cos'], help='distance type of GM')
parser.add_argument('--prune_method', required=True, default='constant', type=str, choices=['constant', 'linear', 'ex'], help='pruning rate method')

args = parser.parse_args()
args.use_cuda = args.ngpu > 0 and torch.cuda.is_available()

if args.manualSeed is None:
    args.manualSeed = random.randint(1, 10000)
random.seed(args.manualSeed)
torch.manual_seed(args.manualSeed)
if args.use_cuda:
    torch.cuda.manual_seed_all(args.manualSeed)
cudnn.benchmark = True

def main():
    # Init logger
    if not os.path.isdir(args.save_path):
        os.makedirs(args.save_path)
    log = open(os.path.join(args.save_path, 'log_seed_{}.txt'.format(args.manualSeed)), 'w')
    print_log('save path : {}'.format(args.save_path), log)
    state = {k: v for k, v in args._get_kwargs()}
    print_log(state, log)
    print_log("Random Seed: {}".format(args.manualSeed), log)
    print_log("python version : {}".format(sys.version.replace('\n', ' ')), log)
    print_log("torch  version : {}".format(torch.__version__), log)
    print_log("cudnn  version : {}".format(torch.backends.cudnn.version()), log)
    print_log("Norm Pruning Rate: {}".format(args.rate_norm), log)
    print_log("Distance Pruning Rate: {}".format(args.rate_dist), log)
    print_log("Layer Begin: {}".format(args.layer_begin), log)
    print_log("Layer End: {}".format(args.layer_end), log)
    print_log("Layer Inter: {}".format(args.layer_inter), log)
    print_log("Epoch prune: {}".format(args.epoch_prune), log)
    print_log("use pretrain: {}".format(args.use_pretrain), log)
    print_log("Pretrain path: {}".format(args.pretrain_path), log)
    print_log("Dist type: {}".format(args.dist_type), log)

    # Init dataset
    if not os.path.isdir(args.data_path):
        os.makedirs(args.data_path)

    if args.dataset == 'cifar10':
        mean = [x / 255 for x in [125.3, 123.0, 113.9]]
        std = [x / 255 for x in [63.0, 62.1, 66.7]]
    elif args.dataset == 'cifar100':
        mean = [x / 255 for x in [129.3, 124.1, 112.4]]
        std = [x / 255 for x in [68.2, 65.4, 70.4]]
    else:
        assert False, "Unknow dataset : {}".format(args.dataset)

    train_transform = transforms.Compose(
        [transforms.RandomHorizontalFlip(), transforms.RandomCrop(32, padding=4), transforms.ToTensor(),
         transforms.Normalize(mean, std)])
    test_transform = transforms.Compose(
        [transforms.ToTensor(), transforms.Normalize(mean, std)])

    if args.dataset == 'cifar10':
        train_data = dset.CIFAR10(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR10(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 10
    elif args.dataset == 'cifar100':
        train_data = dset.CIFAR100(args.data_path, train=True, transform=train_transform, download=True)
        test_data = dset.CIFAR100(args.data_path, train=False, transform=test_transform, download=True)
        num_classes = 100

    train_loader = torch.utils.data.DataLoader(train_data, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.workers, pin_memory=True)


    # Init model, criterion, and optimizer
    net = mobilenetv2.MobileNetV2()
    model_loaded = torch.load('ckpt.pth')
    mobile = model_loaded['net'].copy()
    print(type(mobile))
    net.load_state_dict({k.replace('module.',''):v for k,v in mobile.items()})
    
    print_log("=> network :\n {}".format(net), log)

    net = torch.nn.DataParallel(net, device_ids=list(range(args.ngpu)))

    # define loss function (criterion) and optimizer
    criterion = torch.nn.CrossEntropyLoss()

    optimizer = torch.optim.SGD(net.parameters(), state['lr'], momentum=state['momentum'],
                                weight_decay=state['decay'], nesterov=True)

    if args.use_cuda:
        net.cuda()
        criterion.cuda()

    recorder = RecorderMeter(args.epochs)

    if args.evaluate:
        time1 = time.time()
        validate(test_loader, net, criterion, log)
        time2 = time.time()
        print('function took %0.3f ms' % ((time2 - time1) * 1000.0))
        return

    m = Mask(net)
    m.init_length()
    print("-" * 10 + "one epoch begin" + "-" * 10)
    print("remaining ratio of pruning : Norm is %f" % args.rate_norm)
    print("reducing ratio of pruning : Distance is %f" % args.rate_dist)
    print("total remaining ratio is %f" % (args.rate_norm - args.rate_dist))

    validation_accurate_1, validation_loss_1 = validate(test_loader, net, criterion, log)

    print(" accu before is: %.3f %%" % validation_accurate_1)

    m.model = net

    m.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
    #    m.if_zero()
    m.do_mask()
    m.do_similar_mask()
    net = m.model
    #    m.if_zero()
    if args.use_cuda:
        net = net.cuda()
    validation_accurate_2, validation_loss_2 = validate(test_loader, net, criterion, log)
    print(" accu after is: %s %%" % validation_accurate_2)

    # Main loop
    start_time = time.time()
    epoch_time = AverageMeter()
    small_filter_idx = []
    large_filter_idx = []

    for epoch in range(args.start_epoch, args.epochs):
        current_lr = adjust_lr(optimizer, epoch, args.gammas, args.schedule)

        required_hour, required_mins, required_secs = convert_secs2time(epoch_time.avg * (args.epochs - epoch))
        required_time = '[required: {:02d}:{:02d}:{:02d}]'.format(required_hour, required_mins, required_secs)

        print_log(
            '\n==>>{:s} [Epoch={:03d}/{:03d}] {:s} [lr={:6.4f}]'.format(time_string(), epoch, args.epochs,
                                                                                   required_time, current_lr) \
            + ' [Best : Accuracy={:.2f}, Error={:.2f}]'.format(recorder.max_accuracy(False),
                                                               100 - recorder.max_accuracy(False)), log)

        # train for one epoch
        train_accurate, train_loss = train(train_loader, net, criterion, optimizer, epoch, log, m)

        # evaluate on validation set
#         validation_accurate_1, validation_loss_1 = validate(test_loader, net, criterion, log)
        if epoch % args.epoch_prune == 0 or epoch == args.epochs - 1:
            m.model = net
            m.if_zero()
            m.init_mask(args.rate_norm, args.rate_dist, args.dist_type)
            m.do_mask()
            m.do_similar_mask()
            m.if_zero()
            net = m.model
            if args.use_cuda:
                net = net.cuda()

        validation_accurate_2, validation_loss_2 = validate(test_loader, net, criterion, log)

        best = recorder.update(epoch, train_loss, train_accurate, validation_loss_2, validation_accurate_2)

        save_checkpoint({
            'epoch': epoch + 1,
            'state_dict': net,
            'recorder': recorder,
            'optimizer': optimizer.state_dict(),
        }, best, args.save_path, 'checkpoint.pth.tar')

        # measure elapsed time
        epoch_time.update(time.time() - start_time)
        start_time = time.time()
        recorder.plot_curve(os.path.join(args.save_path, 'curve.png'))

    start = time.time()
    validation_accurate_2, validation_loss_2 = validate(test_loader, net, criterion, log)
    print("Acc=%.4f\n"%(validation_accurate_2))
    print(f"Run time: {(time.time() - start):.3f} s")
    torch.save(net, 'geometric_median-%s%.2f.pth'%(args.prune_method, args.rate_dist))
    log.close()
    

# train function (forward, backward, update)
def train(train_loader, model, criterion, optimizer, epoch, log, m):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()
    # switch to train mode
    model.train()

    end = time.time()
    for i, (input, target) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input)
        target_var = torch.autograd.Variable(target)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        precision1, precision5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(precision1, input.size(0))
        top5.update(precision5, input.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()

        # Mask grad for iteration
        m.do_grad_mask()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print_log('  Epoch: [{:03d}][{:03d}/{:03d}]   '
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})   '
                      'Data {data_time.val:.3f} ({data_time.avg:.3f})   '
                      'Loss {loss.val:.4f} ({loss.avg:.4f})   '
                      'precision@1 {top1.val:.3f} ({top1.avg:.3f})   '
                      'precision@5 {top5.val:.3f} ({top5.avg:.3f})   '.format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses, top1=top1, top5=top5) + time_string(), log)
#     print_log(
#         '  **Train** precision@1 {top1.avg:.3f} precision@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
#                                                                                               error1=100 - top1.avg),
#         log)
    print_log(
            '  **Train** Accuracy@1 {top1.avg:.3f}'.format(top1=top1),log)
    return top1.avg, losses.avg


def validate(val_loader, model, criterion, log):
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()

    for i, (input, target) in enumerate(val_loader):
        if args.use_cuda:
            target = target.cuda(async=True)
            input = input.cuda()
        input_var = torch.autograd.Variable(input, volatile=True)
        target_var = torch.autograd.Variable(target, volatile=True)

        # compute output
        output = model(input_var)
        loss = criterion(output, target_var)

        # measure accuracy and record loss
        precision1, precision5 = accuracy(output.data, target, topk=(1, 5))
        losses.update(loss.data, input.size(0))
        top1.update(precision1, input.size(0))
        top5.update(precision5, input.size(0))

    print_log('  **Test** precision@1 {top1.avg:.3f} precision@5 {top5.avg:.3f} Error@1 {error1:.3f}'.format(top1=top1, top5=top5,
                                                                                                   error1=100 - top1.avg),
              log)

    return top1.avg, losses.avg


def print_log(print_string, log):
    print("{}".format(print_string))
    log.write('{}\n'.format(print_string))
    log.flush()


def save_checkpoint(state, best, save_path, filename):
    filename = os.path.join(save_path, filename)
    torch.save(state, filename)
    if best:
        bestname = os.path.join(save_path, 'model_best.pth.tar')
        shutil.copyfile(filename, bestname)


def adjust_lr(optimizer, epoch, gammas, schedule):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr
    assert len(gammas) == len(schedule), "length of gammas and schedule should be equal"
    for (gamma, step) in zip(gammas, schedule):
        if (epoch >= step):
            lr = lr * gamma
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr
    return lr


def accuracy(output, target, topk=(1,)):
    """Computes the precisionision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res


def save_obj(obj, name):
    with open('obj/' + name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)


def load_obj(name):
    with open('obj/' + name + '.pkl', 'rb') as f:
        return pickle.load(f)


class Mask:
    def __init__(self, model):
        self.model_size = {}
        self.model_length = {}
        self.compress_rate = {}
        self.distance_rate = {}
        self.mat = {}
        self.model = model
        self.mask_idx = []
        self.filter_small_idx = {}
        self.filter_large_idx = {}
        self.similar_matrix = {}
        self.norm_matrix = {}

    def get_codebook(self, weight_torch, compress_rate, length):
        weight_vec = weight_torch.view(length)
        weight_np = weight_vec.cpu().numpy()

        weight_abs = np.abs(weight_np)
        weight_sort = np.sort(weight_abs)

        threshold = weight_sort[int(length * (1 - compress_rate))]
        weight_np[weight_np <= -threshold] = 1
        weight_np[weight_np >= threshold] = 1
        weight_np[weight_np != 1] = 0

        print("codebook done")
        return weight_np

    def get_filter_codebook(self, weight_torch, compress_rate, length):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            norm_2 = torch.norm(weight_vec, 2, 1)
            norm_2_np = norm_2.cpu().numpy()
            filter_idx = norm_2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(filter_idx)):
                codebook[filter_idx[x] * kernel_length: (filter_idx[x] + 1) * kernel_length] = 0

#             print("filter codebook done")
        else:
            pass
        return codebook

    def get_filter_idx(self, weight_torch, compress_rate, length):
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)
            # norm1 = torch.norm(weight_vec, 1, 1)
            # norm1_np = norm1.cpu().numpy()
            norm_2 = torch.norm(weight_vec, 2, 1)
            norm_2_np = norm_2.cpu().numpy()
            filter_small_idx = []
            filter_large_idx = []
            filter_large_idx = norm_2_np.argsort()[filter_pruned_num:]
            filter_small_idx = norm_2_np.argsort()[:filter_pruned_num]
            #            norm1_sort = np.sort(norm1_np)
            #            threshold = norm1_sort[int (weight_torch.size()[0] * (1-compress_rate) )]
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            # print("filter idx done")
        else:
            pass
        return filter_small_idx, filter_large_idx

    # optimize for fast ccalculation
    def get_filter_similar(self, weight_torch, compress_rate, distance_rate, length, dist_type="l2"):
        codebook = np.ones(length)
        if len(weight_torch.size()) == 4:
            filter_pruned_num = int(weight_torch.size()[0] * (1 - compress_rate))
            similar_pruned_num = int(weight_torch.size()[0] * distance_rate)
            weight_vec = weight_torch.view(weight_torch.size()[0], -1)

            if dist_type == "l2" or "cos":
                norm = torch.norm(weight_vec, 2, 1)
                norm_np = norm.cpu().numpy()
            elif dist_type == "l1":
                norm = torch.norm(weight_vec, 1, 1)
                norm_np = norm.cpu().numpy()
            filter_small_idx = []
            filter_large_idx = []
            filter_large_idx = norm_np.argsort()[filter_pruned_num:]
            filter_small_idx = norm_np.argsort()[:filter_pruned_num]

            # # distance using pytorch function
            # similar_matrix = torch.zeros((len(filter_large_idx), len(filter_large_idx)))
            # for x1, x2 in enumerate(filter_large_idx):
            #     for y1, y2 in enumerate(filter_large_idx):
            #         # cos = torch.nn.CosineSimilarity(dim=1, eps=1e-6)
            #         # similar_matrix[x1, y1] = cos(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0]
            #         pdist = torch.nn.PairwiseDistance(p=2)
            #         similar_matrix[x1, y1] = pdist(weight_vec[x2].view(1, -1), weight_vec[y2].view(1, -1))[0][0]
            # # more similar with other filter indicates large in the sum of row
            # similar_sum = torch.sum(torch.abs(similar_matrix), 0).numpy()

            # distance using numpy function
            indices = torch.LongTensor(filter_large_idx).cuda()
            weight_after_norm = torch.index_select(weight_vec, 0, indices).cpu().numpy()
            # for euclidean distance
            if dist_type == "l2" or "l1":
                similar_matrix = distance.cdist(weight_after_norm, weight_after_norm, 'euclidean')
            elif dist_type == "cos":  # for cos similarity
                similar_matrix = 1 - distance.cdist(weight_after_norm, weight_after_norm, 'cosine')
            similar_sum = np.sum(np.abs(similar_matrix), axis=0)

            # for distance similar: get the filter idx with largest similarity == small distance
            similar_large_idx = similar_sum.argsort()[similar_pruned_num:]
            similar_small_idx = similar_sum.argsort()[:  similar_pruned_num]
            similar_idx_for_filter = [filter_large_idx[i] for i in similar_small_idx]

#             print('filter_large_idx', filter_large_idx)
#             print('filter_small_idx', filter_small_idx)
#             print('similar_sum', similar_sum)
#             print('similar_large_idx', similar_large_idx)
#             print('similar_small_idx', similar_small_idx)
#             print('similar_idx_for_filter', similar_idx_for_filter)
            kernel_length = weight_torch.size()[1] * weight_torch.size()[2] * weight_torch.size()[3]
            for x in range(0, len(similar_idx_for_filter)):
                codebook[
                similar_idx_for_filter[x] * kernel_length: (similar_idx_for_filter[x] + 1) * kernel_length] = 0
#             print("similar idx done")
        else:
            pass
        return codebook

    def convert2tensor(self, x):
        x = torch.FloatTensor(x)
        return x

    def init_length(self):
        for idx, item in enumerate(self.model.parameters()):
            self.model_size[idx] = item.size()

        for idx1 in self.model_size:
            for idx2 in range(0, len(self.model_size[idx1])):
                if idx2 == 0:
                    self.model_length[idx1] = self.model_size[idx1][0]
                else:
                    self.model_length[idx1] *= self.model_size[idx1][idx2]

    def init_rate(self, rate_norm_per_layer, rate_dist_per_layer):
        for idx, item in enumerate(self.model.parameters()):
            self.compress_rate[idx] = 1
            self.distance_rate[idx] = 1
        for key in range(args.layer_begin, args.layer_end + 1, args.layer_inter):
            self.compress_rate[key] = rate_norm_per_layer
            self.distance_rate[key] = rate_dist_per_layer
        n = rate_dist_per_layer
        p_rate_list_constant = [n]*17
        p_rate_list_linear = [0,n,n,n*2,n*2,n*2,n*3,n*3,n*3,n*3,n*4,n*4,n*4,n*5,n*5,n*5,n*6]
        n = 1 - rate_dist_per_layer # remaining rate
        r_rate_list = [1,n,n,n**2,n**2,n**2,n**3,n**3,n**3,n**3,n**4,n**4,n**4,n**5,n**5,n**5,n**6]
        p_rate_list_ex = [(1-x) for x in r_rate_list]
        counter = 0
        counter2 = 0
        self.mask_idx = []
        for name, param in self.model.named_parameters():
            if 'module.layers.' in name and ('conv1' in name or 'conv2' in name):
                self.mask_idx.append(counter)
                if args.prune_method == 'constant':
                    self.distance_rate[counter] = p_rate_list_constant[int(counter2)]
                elif args.prune_method == 'linear':
                    self.distance_rate[counter] = p_rate_list_linear[int(counter2)]
                elif args.prune_method == 'ex':
                    self.distance_rate[counter] = p_rate_list_ex[int(counter2)]
                counter2 += 0.5
            counter += 1       

    def init_mask(self, rate_norm_per_layer, rate_dist_per_layer, dist_type):
        self.init_rate(rate_norm_per_layer, rate_dist_per_layer)
        counter = 0
        for idx, item in enumerate(self.model.parameters()):
#             print(idx)
            if idx in self.mask_idx:
#                 print('***************************************%d***************************************'%idx)
#                 print(item)
                counter += 1
                # mask for norm criterion
                self.mat[idx] = self.get_filter_codebook(item.data, self.compress_rate[idx],
                                                           self.model_length[idx])
                self.mat[idx] = self.convert2tensor(self.mat[idx])
                if args.use_cuda:
                    self.mat[idx] = self.mat[idx].cuda()

                # # get result about filter idx
                # self.filter_small_idx[idx], self.filter_large_idx[idx] = \
                #     self.get_filter_idx(item.data, self.compress_rate[idx], self.model_length[idx])

                # mask for distance criterion
                self.similar_matrix[idx] = self.get_filter_similar(item.data, self.compress_rate[idx],
                                                                     self.distance_rate[idx],
                                                                     self.model_length[idx], dist_type=dist_type)
                self.similar_matrix[idx] = self.convert2tensor(self.similar_matrix[idx])
                if args.use_cuda:
                    self.similar_matrix[idx] = self.similar_matrix[idx].cuda()
        print("mask Ready")

    def do_mask(self):
        for idx, item in enumerate(self.model.parameters()):
            if idx in self.mask_idx:
                a = item.data.view(self.model_length[idx])
                b = a * self.mat[idx]
                item.data = b.view(self.model_size[idx])
        print("mask Done")

    def do_similar_mask(self):
        for idx, item in enumerate(self.model.parameters()):
            if idx in self.mask_idx:
                a = item.data.view(self.model_length[idx])
                b = a * self.similar_matrix[idx]
                item.data = b.view(self.model_size[idx])
        print("mask similar Done")

    def do_grad_mask(self):
        for idx, item in enumerate(self.model.parameters()):
            if idx in self.mask_idx:
                a = item.grad.data.view(self.model_length[idx])
                # reverse the mask of model
                # b = a * (1 - self.mat[idx])
                b = a * self.mat[idx]
                b = b * self.similar_matrix[idx]
                item.grad.data = b.view(self.model_size[idx])
        # print("grad zero Done")

    def if_zero(self):
        for idx, item in enumerate(self.model.parameters()):
            if (idx in self.mask_idx):
                # if idx == 0:
                a = item.data.view(self.model_length[idx])
                b = a.cpu().numpy()

                print(
                    "number of nonzero weight is %d, zero is %d" % (np.count_nonzero(b), len(b) - np.count_nonzero(b)))


if __name__ == '__main__':
    main()

