# encoding: utf-8
"""
Training implementation for CIFAR100 dataset  
Author: Jason.Fang
Update time: 15/07/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import numpy as np
import pandas as pd
import torch
import torchvision
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
import math
from thop import profile
from tensorboardX import SummaryWriter
#define by myself
from utils.common import count_bytes
from nets.resnet import resnet18
from nets.densenet import densenet121

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
max_epoches = 50
batch_size = 256
CKPT_PATH = '/data/pycode/LungCT3D/ckpt/cifar_resnet_conv_mf.pkl'
#https://pytorch.org/tutorials/beginner/blitz/cifar10_tutorial.html
def Train():
    print('********************load data********************')
    root = '/data/tmpexec/cifar'
    if not os.path.exists(root):
        os.mkdir(root)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # if not exist, download mnist dataset
    train_set = dset.CIFAR10(root=root, train=True, transform=trans, download=True)
    test_set = dset.CIFAR10(root=root, train=False, transform=trans, download=True)
    #classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    #split train set and val set
    sample_size = int(1.0 * len(train_set)/5) #[1.0, 1/5]
    train_set, _ = torch.utils.data.random_split(train_set, [sample_size, len(train_set) - sample_size])
    train_size = int(0.8 * len(train_set))#8:2
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    batch_size=batch_size,
                    shuffle=True, num_workers=8)
    val_loader = torch.utils.data.DataLoader(
                    dataset=val_dataset,
                    batch_size=batch_size,
                    shuffle=False, num_workers=8)

    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total validation batch number: {}'.format(len(val_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = resnet18(pretrained=False, num_classes=100)
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    criterion = nn.CrossEntropyLoss().cuda()
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    acc_min = 0.30 #float('inf')
    for epoch in range(max_epoches):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , max_epoches))
        print('-' * 10)
        model.train()  #set model to training mode
        loss_train = []
        with torch.autograd.enable_grad():
            for batch_idx, (img, lbl) in enumerate(train_loader):
                #forward
                var_image = torch.autograd.Variable(img).cuda()
                var_label = torch.autograd.Variable(lbl).cuda()
                var_out = model(var_image)
                # backward and update parameters
                optimizer_model.zero_grad()
                loss_tensor = criterion.forward(var_out, var_label) 
                loss_tensor.backward()
                optimizer_model.step()
                #show 
                loss_train.append(loss_tensor.item())
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item()) ))
                sys.stdout.flush()
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(loss_train) ))

        #test
        model.eval()
        loss_test = []
        total_cnt, correct_cnt = 0, 0
        with torch.autograd.no_grad():
            for batch_idx,  (img, lbl) in enumerate(val_loader):
                #forward
                var_image = torch.autograd.Variable(img).cuda()
                var_label = torch.autograd.Variable(lbl).cuda()
                var_out = model(var_image)
                loss_tensor = criterion.forward(var_out, var_label)
                loss_test.append(loss_tensor.item())
                _, pred_label = torch.max(var_out.data, 1)
                total_cnt += var_image.data.size()[0]
                correct_cnt += (pred_label == var_label.data).sum()
                sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
                sys.stdout.flush()
        acc = correct_cnt * 1.0 / total_cnt
        print("\r Eopch: %5d val loss = %.6f, ACC = %.6f" % (epoch + 1, np.mean(loss_test), acc) )

        # save checkpoint
        if acc_min < acc:
            acc_min = acc
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    root = '/data/tmpexec/cifar'
    if not os.path.exists(root):
        os.mkdir(root)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

    # if not exist, download mnist dataset
    train_set = dset.CIFAR10(root=root, train=True, transform=trans, download=True)
    test_set = dset.CIFAR10(root=root, train=False, transform=trans, download=True)
    classes = ('plane', 'car', 'bird', 'cat','deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_set,
                    batch_size=batch_size,
                    shuffle=True, num_workers=8)
    test_loader = torch.utils.data.DataLoader(
                    dataset=test_set,
                    batch_size=batch_size,
                    shuffle=False, num_workers=8)

    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total testing batch number: {}'.format(len(test_loader)))
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = resnet18(pretrained=False, num_classes=100).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model.eval()#turn to test mode
    #param = sum(p.numel() for p in model.parameters() if p.requires_grad) #count params of model
    print('********************load model succeed!********************')

    print('********************begin Testing!********************')
    total_cnt, top1, top5 = 0, 0, 0
    time_res = []
    with torch.autograd.no_grad():
        for batch_idx,  (img, lbl) in enumerate(test_loader):
            #forward
            var_image = torch.autograd.Variable(img).cuda()
            var_label = torch.autograd.Variable(lbl).cuda()
            start = time.time()
            var_out = model(var_image)
            end = time.time()
            time_res.append(end-start)

            total_cnt += var_image.data.size()[0]
            _, pred_label = torch.max(var_out.data, 1) #top1
            top1 += (pred_label == var_label.data).sum()
            _, pred_label = torch.topk(var_out.data, 5, 1)#top5
            pred_label = pred_label.t()
            pred_label = pred_label.eq(var_label.data.view(1, -1).expand_as(pred_label))
            top5 += pred_label.float().sum()

            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    
    param = sum(p.numel() for p in model.parameters() if p.requires_grad) #count params of model
    print("\r Params of model: {}".format(count_bytes(param)) )
    flops, _ = profile(model, inputs=(var_image,))
    print("FLOPs(Floating Point Operations) of model = {}".format(count_bytes(flops)) )
    print("FPS(Frams Per Second) of model = %.2f"% (1.0/(np.sum(time_res)/len(time_res))) )

    acc = top1 * 1.0 / total_cnt
    ci  = 1.96 * math.sqrt( (acc * (1 - acc)) / total_cnt) #1.96-95%
    print("\r Top-1 ACC/CI = %.4f/%.4f" % (acc, ci) )
    acc = top5 * 1.0 / total_cnt
    ci  = 1.96 * math.sqrt( (acc * (1 - acc)) / total_cnt) #1.96-95%
    print("\r Top-5 ACC/CI = %.4f/%.4f" % (acc, ci) )

def TensorboardDemo():

    print('********************load data********************')
    root = '/data/tmpexec/fashion-mnist'
    if not os.path.exists(root):
        os.mkdir(root)
    trans = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5,), (0.5,))])
    # if not exist, download fashion mnist dataset
    train_set = dset.FashionMNIST(root=root, train=True, transform=trans, download=True)
    test_set = dset.FashionMNIST(root=root, train=False, transform=trans, download=True)

    #split train set and val set
    sample_size = int(1.0 * len(train_set)/6) #[1.0, 1/6]
    train_set, _ = torch.utils.data.random_split(train_set, [sample_size, len(train_set) - sample_size])
    train_size = int(0.8 * len(train_set))#8:2
    val_size = len(train_set) - train_size
    train_dataset, val_dataset = torch.utils.data.random_split(train_set, [train_size, val_size])

    train_loader = torch.utils.data.DataLoader(
                    dataset=train_dataset,
                    batch_size=4,
                    shuffle=True, num_workers=1)
    val_loader = torch.utils.data.DataLoader(
                    dataset=val_dataset,
                    batch_size=4,
                    shuffle=False, num_workers=1)

    print ('==>>> total trainning batch number: {}'.format(len(train_loader)))
    print ('==>>> total validation batch number: {}'.format(len(val_loader)))
    # constant for classes
    classes = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = resnet18(pretrained=False, num_classes=10).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    criterion = nn.CrossEntropyLoss()
    print('********************load model succeed!********************')

    print('********************Visualization with Tensorboard!********************')
    writer = SummaryWriter('/data/tmpexec/tensorboard-log') #--port 10002
    # get some random training images
    dataiter = iter(train_loader)
    images, labels = dataiter.next()
    # create grid of images
    img_grid = torchvision.utils.make_grid(images)
    # show images
    matplotlib_imshow(img_grid, one_channel=False)
    # write to tensorboard
    writer.add_image('fashion', img_grid)

    writer.add_graph(model, images.cuda())
    writer.close()


def matplotlib_imshow(img, one_channel=False):
    if one_channel:
        img = img.mean(dim=0)
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    if one_channel:
        plt.imshow(npimg, cmap="Greys")
    else:
        plt.imshow(np.transpose(npimg, (1, 2, 0)))

def main():
    #Train()
    #Test()
    TensorboardDemo()

if __name__ == '__main__':
    main()
