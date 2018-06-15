
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

from torchvision import datasets, transforms
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

import ipdb as pdb


'''
Save the model for later
'''
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count



def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = args.lr * (0.1 ** (epoch // 30))
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr



def adjust_learning_rate2(optimizer):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    lr = float(args.lr) / 4.0
    for param_group in optimizer.param_groups:
        param_group['lr'] = lr


# TODO: Repair the accuracy function


def accuracy(output, target,topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    target = target.view(target.size(0)).long()
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].view(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res







def visualize(data, fig_num, title):
    input_tensor = data.cpu()
    input_tensor = torch.squeeze(input_tensor)
    in_grid = input_tensor.detach().numpy()

    fig=plt.figure(num = fig_num)
    plt.imshow(in_grid, cmap='gray', interpolation='none')
    plt.title(title)
    figManager = plt.get_current_fig_manager()
    figManager.resize(*figManager.window.maxsize())
    plt.show(block=False)
    # time.sleep(4)
    # plt.close()



def visualize_stn(data, title):
    input_tensor = data.cpu()
    input_tensor = torch.squeeze(input_tensor)
    input_tensor = input_tensor.detach().numpy()
    N = len(input_tensor)

    fig=plt.figure(num = 2)

    columns = N
    rows = 1
    for i in range(1, columns*rows +1):
        img = input_tensor[i-1]
        fig.add_subplot(rows, columns, i)
        plt.imshow(img, cmap='gray', interpolation='none')

    figManager = plt.get_current_fig_manager()
    figManager.resize(*figManager.window.maxsize())
    # figManager.window.state('zoomed')
    plt.show(block=False)
    # time.sleep(4)
    # plt.close()
