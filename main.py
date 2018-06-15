#!/usr/bin/env python

"""
Author: Anshul Paigwar
email: p.anshul6@gmail.com
"""



from __future__ import print_function

import argparse
import os
import shutil
import time

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
from torch.autograd import Variable
import matplotlib
import matplotlib.pyplot as plt

import numpy as np
from modules import EDRAM_Loss
from model_stn import RecurrentSpatialTransformer
from tools.mnist_seq_dataset import get_data_loaders
from tools.utils import save_checkpoint, AverageMeter, accuracy, visualize,visualize_stn

from torchviz import make_dot
import ipdb as pdb



use_cuda = torch.cuda.is_available()




parser = argparse.ArgumentParser()

# specify data and datapath
parser.add_argument('--dataset',  default='modelnet40_pcl', help='modelnet40_pcl | ?? ')
parser.add_argument('--data_dir', default="/home/anshul/inria_thesis/datasets/modelnet40_ply_hdf5_2048", help='path to dataset')
parser.add_argument('--batchSize', type=int, default=32, help='input batch size')
parser.add_argument('--lr', '--learning-rate', default=20, type=float, help='initial learning rate')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-e', '--evaluate', dest='evaluate', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--pretrained', dest='pretrained', action='store_true',
                    help='use pre-trained model')
parser.add_argument('-s', '--save_checkpoints', dest='save_checkpoints', action='store_true',
                    help='evaluate model on validation set')
parser.add_argument('--epochs', default=90, type=int,
                    help='number of total epochs to run')
parser.add_argument('--num_glimpses', default=6, type=int,
                    help='number of total epochs to run')
parser.add_argument('--start_epoch', default=0, type=int,
                    help='number  epochs to start from')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')



args = parser.parse_args()






data_dir = "/home/anshul/inria_thesis/datasets/mnist_sequence3_sample_8distortions_9x9/"
train_loader, valid_loader = get_data_loaders(data_dir, batch = args.batchSize, transform = True)


model = RecurrentSpatialTransformer() # FIXME input parameters
if use_cuda:
    print("using cuda")
    model.cuda()

optimizer = optim.SGD(model.parameters(), lr=0.01)
# optimizer = optim.Adam(model.parameters(), args.lr)

# loss_func_1 = EDRAM_Loss().cuda()
criterion = nn.CrossEntropyLoss()


# def criterion(output,loc_estimate_list, target, loc_true):
#     # return loss_func_1(output,loc_estimate_list, target, loc_true)
#     return loss_func_2(output,target)







def train_stn(epoch):

    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()
    end = time.time()

    for batch_idx, (data, data_resized, target) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        B = data.shape[0] # Batch size
        if use_cuda:
            data, data_resized, target = data.cuda(),data_resized.cuda(), target.cuda()
        data, data_resized = data.unsqueeze(1).float(), data_resized.unsqueeze(1).float()

        optimizer.zero_grad()

        # hidden = torch.rand(1,B,1024).cuda()
        hidden = model.context_2(data_resized)
        hidden = hidden.unsqueeze(0)
        target = target.long()


        total_loss = []
        loss = None
        seq_len = 3

        output, glimpse_list, loc_estimate_list = model(data,hidden,seq_len)
        for i in range(seq_len):
            loss = criterion(output[i], target[:,i])
            # loss = criterion(output[i],loc_estimate_list[i], target, loc_true)
            total_loss.append(loss)
        total_loss = torch.stack(total_loss).mean()

        # measure accuracy and record loss
        prec1, prec5 = accuracy(output[seq_len-1], target[:, seq_len-1], topk=(1, 5))
        losses.update(total_loss.item(), B)
        # losses.update(total_loss.item(), B)
        top1.update(prec1[0], B)
        top5.update(prec5[0], B)

        # compute gradient and do SGD step
        total_loss.backward()
        # total_loss.backward()
        optimizer.step()

        # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
        hidden = hidden.detach()


        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # print(glimpse_list[1])
        # pdb.set_trace()



        if batch_idx % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                   epoch, batch_idx, len(train_loader), batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, top5=top5))








def validate_stn():
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to evaluate mode
    model.eval()
    with torch.no_grad():
        end = time.time()

        for batch_idx, (data, data_resized, target) in enumerate(valid_loader):

            B = data.shape[0] # Batch size
            if use_cuda:
                data, data_resized, target = data.cuda(),data_resized.cuda(), target.cuda()
            data, data_resized = data.unsqueeze(1).float(), data_resized.unsqueeze(1).float()

            optimizer.zero_grad()

            # hidden = torch.rand(1,B,1024).cuda()
            hidden = model.context_2(data_resized)
            hidden = hidden.unsqueeze(0)
            target = target.long()


            total_loss = []
            loss = None
            seq_len = 3

            output, glimpse_list, loc_estimate_list = model(data,hidden,seq_len)
            for i in range(seq_len):
                loss = criterion(output[i], target[:,i])
                # loss = criterion(output[i],loc_estimate_list[i], target, loc_true)
                total_loss.append(loss)
            total_loss = torch.stack(total_loss).mean()

            # measure accuracy and record loss
            prec1, prec5 = accuracy(output[seq_len-1], target[:, seq_len-1], topk=(1, 5))
            losses.update(total_loss.item(), B)
            # losses.update(total_loss.item(), B)
            top1.update(prec1[0], B)
            top5.update(prec5[0], B)

            torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
            hidden = hidden.detach()


            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            # print('where')

            if batch_idx % args.print_freq == 0:
                print('Test: [{0}/{1}]\t'
                      'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                      'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                      'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                      'Prec@5 {top5.val:.3f} ({top5.avg:.3f})'.format(
                       batch_idx, len(valid_loader), batch_time=batch_time, loss=losses,
                       top1=top1, top5=top5))

            if batch_idx % 1400 == 0:
                plt.close()
                print("target", target[1])
                visualize(data[1], 1, "dataset image")
                # visualize(glimpse_list, 2, "Transformed Images")


                # stn_check = stn_zoom(out_height=26, out_width=26)
                # true_attention = stn_check(data,target_loc[1])
                # visualize(true_attention[1], "GroundTruth Attention")

                visualize_stn(glimpse_list, "Transformed Images")

        print(' * Prec@1 {top1.avg:.3f} Prec@5 {top5.avg:.3f}'
              .format(top1=top1, top5=top5))

    return top1.avg





best_prec1 = 0

def main():
    global args, best_prec1
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            # args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    if args.evaluate:
        validate(val_loader, model, criterion)
        return


    for epoch in range(args.epochs):
        # adjust_learning_rate(optimizer, epoch)
        train_stn(epoch)
        # evaluate on validation set
        prec1 = validate_stn()

        # if (prec1 < best_prec1):
        #     adjust_learning_rate2(optimizer)

        if args.save_checkpoints:
            # remember best prec@1 and save checkpoint
            is_best = prec1 > best_prec1
            best_prec1 = max(prec1, best_prec1)
            save_checkpoint({
                'epoch': epoch + 1,
                'state_dict': model.state_dict(),
                'best_prec1': best_prec1,
                'optimizer' : optimizer.state_dict(),
            }, is_best)


if __name__ == '__main__':
    main()





# '''
# Save the model for later
# '''
# def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
#     torch.save(state, filename)
#     if is_best:
#         shutil.copyfile(filename, 'model_best.pth.tar')
#
#
#
# class AverageMeter(object):
#     """Computes and stores the average and current value"""
#     def __init__(self):
#         self.reset()
#
#     def reset(self):
#         self.val = 0
#         self.avg = 0
#         self.sum = 0
#         self.count = 0
#
#     def update(self, val, n=1):
#         self.val = val
#         self.sum += val * n
#         self.count += n
#         self.avg = self.sum / self.count
#
#
#
# def adjust_learning_rate(optimizer, epoch):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = args.lr * (0.1 ** (epoch // 30))
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#
#
# def adjust_learning_rate2(optimizer):
#     """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
#     lr = float(args.lr) / 4.0
#     for param_group in optimizer.param_groups:
#         param_group['lr'] = lr
#
#
# # TODO: Repair the accuracy function
#
#
# def accuracy(output, target,topk=(1,)):
#     """Computes the precision@k for the specified values of k"""
#     target = target.view(target.size(0)).long()
#     with torch.no_grad():
#         maxk = max(topk)
#         batch_size = target.size(0)
#
#         _, pred = output.topk(maxk, 1, True, True)
#         pred = pred.t()
#         correct = pred.eq(target.view(1, -1).expand_as(pred))
#
#         res = []
#         for k in topk:
#             correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
#             res.append(correct_k.mul_(100.0 / batch_size))
#         return res
#
#
#
#
#
# def convert_image_np(inp):
#     """Convert a Tensor to numpy image."""
#     inp = inp.detach().numpy().transpose((1, 2, 0))
#     # mean = np.array([0.485, 0.456, 0.406])
#     # std = np.array([0.229, 0.224, 0.225])
#     # inp = std * inp + mean
#     # inp = np.clip(inp, 0, 1)
#     return inp
#
#
#
#
#
#
# def visualize(data, fig_num, title):
#     input_tensor = data.cpu()
#     input_tensor = torch.squeeze(input_tensor)
#     in_grid = input_tensor.detach().numpy()
#     # in_grid = convert_image_np((input_tensor))
#     # Plot the results side-by-side
#     fig=plt.figure(num = fig_num)
#     plt.imshow(in_grid, cmap='gray', interpolation='none')
#     plt.title(title)
#     # plt.switch_backend('TkAgg')
#     figManager = plt.get_current_fig_manager()
#     figManager.resize(*figManager.window.maxsize())
#     # figManager.window.state('zoomed')
#     plt.show(block=False)
#     # time.sleep(4)
#     # plt.close()
#
#
#
# def visualize_stn(data, title):
#     input_tensor = data.cpu()
#     input_tensor = torch.squeeze(input_tensor)
#     input_tensor = input_tensor.detach().numpy()
#     # print(input_tensor.shape)
#     N = len(input_tensor)
#
#     fig=plt.figure(num = 2)
#     fig=plt.figure(figsize=(8, 8))
#     columns = N
#     rows = 1
#     for i in range(1, columns*rows +1):
#         img = input_tensor[i-1]
#         # img = convert_image_np(input_tensor[i-1])
#         fig.add_subplot(rows, columns, i)
#         plt.imshow(img, cmap='gray', interpolation='none')
#     # plt.switch_backend('TkAgg')
#     figManager = plt.get_current_fig_manager()
#     figManager.resize(*figManager.window.maxsize())
#     # figManager.window.state('zoomed')
#     plt.show(block=False)
#     # time.sleep(4)
#     # plt.close()
#
#
#
#
#
#
#
#
#
#
#
# def cal_accuracy(output, target):
#     target = target.view(target.size(0)).long()
#     with torch.no_grad():
#         batch_size = target.size(0)
#         _, predicted = torch.max(outputs.data, 1)
#         correct = (predicted == target).sum().item()
#         return correct * 100.0 / batch_size
#
