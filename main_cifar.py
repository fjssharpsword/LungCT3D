# encoding: utf-8
"""
Training implementation for CIFAR10 dataset  
Author: Jason.Fang
Update time: 07/07/2021
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
import torch.nn as nn
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torch.nn.functional as F
import torch.optim as optim
from torch.optim import lr_scheduler
import matplotlib.pyplot as plt
#define by myself
from nets.dml_2dnet_res import bayes_resnet18
#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
max_epoches = 20
batch_size = 320
CKPT_PATH = '/data/pycode/LungCT3D/ckpt/resnet_cifar_best.pkl'
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
    model = bayes_resnet18(num_classes=10)
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
    loss_min = float('inf')
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
            for batch_idx,  (img, lbl) in enumerate(train_loader):
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
        print("\r Eopch: %5d test loss = %.6f, ACC = %.6f" % (epoch + 1, np.mean(loss_test), correct_cnt * 1.0 / total_cnt) )

        # save checkpoint
        if loss_min > np.mean(loss_test):
            loss_min = np.mean(loss_test)
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch + 1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    pass

def main():
    Train()
    Test()

if __name__ == '__main__':
    main()
