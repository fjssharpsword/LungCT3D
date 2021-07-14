# encoding: utf-8
"""
Training implementation for VIN-CXR dataset - Segmentation - 2D UNet
Author: Jason.Fang
Update time: 12/07/2021
"""
import re
import sys
import os
import cv2
import time
import argparse
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
import torch.nn.functional as F
from skimage.measure import label
from sklearn.metrics import roc_auc_score, roc_curve, auc, f1_score, confusion_matrix
from sklearn.metrics.pairwise import cosine_similarity
import matplotlib.pyplot as plt
from thop import profile
#define by myself
from utils.common import compute_AUCs, count_bytes
from data_cxr2d.vincxr_dataloader import get_train_dataloader_VIN, get_test_dataloader_VIN
from nets.unet2d import UNet

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "0,1,2,3,4,5,6,7"
CLASS_NAMES_Vin = ['No finding', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
        'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
BATCH_SIZE = 32
MAX_EPOCHS = 20
NUM_CLASSES =  len(CLASS_NAMES_Vin)
CKPT_PATH = '/data/pycode/LungCT3D/ckpt/unet2d_best.pkl'
def Train():
    print('********************load data********************')
    dataloader_train = get_train_dataloader_VIN(batch_size=BATCH_SIZE, shuffle=True, num_workers=8)
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = UNet(n_channels=3, n_classes=1)
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained segmentation model checkpoint of Vin-CXR dataset: "+CKPT_PATH)
    model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    criterion = nn.BCELoss() #nn.CrossEntropyLoss()
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    loss_min = float('inf')
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (image, _, mask) in enumerate(dataloader_train):
                var_image = torch.autograd.Variable(image).cuda()
                var_mask = torch.autograd.Variable(mask).cuda()
                var_out = model(var_image)
                loss_tensor = criterion(var_out, var_mask)

                optimizer_model.zero_grad()
                loss_tensor.backward()
                optimizer_model.step()#update parameters
                
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item())))
                sys.stdout.flush()
                train_loss.append(loss_tensor.item())
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        #save checkpoint with lowest loss 
        if loss_min > np.mean(train_loss):
            loss_min = np.mean(train_loss)
            torch.save(model.module.state_dict(), CKPT_PATH) #Saving torch.nn.DataParallel Models
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))

def Test():
    print('********************load data********************')
    dataloader_test = get_test_dataloader_VIN(batch_size=32, shuffle=False, num_workers=8) #BATCH_SIZE
    print('********************load data succeed!********************')

    print('********************load model********************')
    model = UNet(n_channels=3, n_classes=1).cuda()
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained segmentation model checkpoint of Vin-CXR dataset: "+CKPT_PATH)
    model.eval()
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    time_res = []
    with torch.autograd.no_grad():
        for batch_idx, (image, _, mask) in enumerate(dataloader_test):
            var_image = torch.autograd.Variable(image).cuda()
            start = time.time()
            var_out = model(var_image)
            end = time.time()
            time_res.append(end-start)
            pred = torch.cat((pred, var_out.data.cpu()), 0) #prob
            gt = torch.cat((gt, mask), 0)
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    #metric
    gt_np = gt.numpy()
    pred_np = np.where(pred.numpy() > 0.5, 1, 0)#pred.numpy()
    # Compute Dice coefficient
    intersection = np.logical_and(gt_np, pred_np)
    dice_coe = 2. * intersection.sum() / (gt_np.sum() + pred_np.sum())
    print("\r Dice coefficient = %.4f" % (dice_coe))
    #model
    print("FPS(Frams Per Second) of model = %.2f"% (1.0/(np.sum(time_res)/len(time_res))) )
    param = sum(p.numel() for p in model.parameters() if p.requires_grad) #count params of model
    print("\r Params of model: {}".format(count_bytes(param)) )
    flops, _ = profile(model, inputs=(var_image,))
    print("FLOPs(Floating Point Operations) of model = {}".format(count_bytes(flops)) )

def main():
    Train()
    Test()

if __name__ == '__main__':
    main()