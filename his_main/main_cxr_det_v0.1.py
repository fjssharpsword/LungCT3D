# encoding: utf-8
"""
Training implementation of object detection for 2D chest x-ray
Author: Jason.Fang
Update time: 19/07/2021
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
from torch.optim import lr_scheduler
import torch.optim as optim
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from tensorboardX import SummaryWriter
from thop import profile
from sklearn.metrics import roc_auc_score
#define by myself
from utils.common import compute_iou, count_bytes
from data_cxr2d.vincxr_coco import get_box_dataloader_VIN
from data_cxr2d.CVTECXR_Test import get_dataloader_CVTE
from nets.resnet import resnet18
from nets.densenet import densenet121

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "6"
CLASS_NAMES_Vin = ['Average', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
        'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
BACKBONE_PARAMS = ['4.0.conv1.weight', '4.0.conv1.module.weight', '4.0.conv1.module.weight_p', '4.0.conv1.module.weight_q',\
                   '5.0.conv1.weight','5.0.conv1.module.weight', '5.0.conv1.module.weight_p', '5.0.conv1.module.weight_q', \
                  '6.0.conv1.weight','6.0.conv1.module.weight', '6.0.conv1.module.weight_p', '6.0.conv1.module.weight_q', \
                  '7.0.conv1.weight','7.0.conv1.module.weight', '7.0.conv1.module.weight_p', '7.0.conv1.module.weight_q' ]
BATCH_SIZE = 16
MAX_EPOCHS = 20
NUM_CLASSES =  len(CLASS_NAMES_Vin)
CKPT_PATH = '/data/pycode/LungCT3D/ckpt/vincxr_densenet_conv_mf.pkl'

def Train(data_loader_box_train):
    print('********************load model********************')
    #resnet = resnet18(pretrained=False, num_classes=NUM_CLASSES).cuda()
    #backbone = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4)
    backbone = densenet121(pretrained=False, num_classes=NUM_CLASSES).features.cuda()
    backbone.out_channels = 1024 #resnet18=512,  densenet121=1024
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=NUM_CLASSES, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler).cuda()
    
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    #model = nn.DataParallel(model).cuda()  # make model available multi GPU cores training    
    torch.backends.cudnn.benchmark = True  # improve train speed slightly
    optimizer_model = optim.Adam(model.parameters(), lr=1e-3, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-5)
    lr_scheduler_model = lr_scheduler.StepLR(optimizer_model , step_size = 10, gamma = 1)
    print('********************load model succeed!********************')

    print('********************begin training!********************')
    #log_writer = SummaryWriter('/data/tmpexec/tensorboard-log') #--port 10002, start tensorboard
    loss_min = float('inf')
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (images, targets) in enumerate(data_loader_box_train):
                optimizer_model.zero_grad()
                images = list(image.cuda() for image in images)
                targets = [{k:v.squeeze(0).cuda() for k, v in t.items()} for t in targets]
                loss_dict  = model(images,targets)   # Returns losses and detections
                loss_tensor = sum(loss for loss in loss_dict.values())
                loss_tensor.backward()
                optimizer_model.step()##update parameters
                sys.stdout.write('\r Epoch: {} / Step: {} : train loss = {}'.format(epoch+1, batch_idx+1, float('%0.6f'%loss_tensor.item())))
                sys.stdout.flush()
                train_loss.append(loss_tensor.item())
        lr_scheduler_model.step()  #about lr and gamma
        print("\r Eopch: %5d train loss = %.6f" % (epoch + 1, np.mean(train_loss))) 

        if loss_min > np.mean(train_loss):
            loss_min = np.mean(train_loss)
            torch.save(model.state_dict(), CKPT_PATH) #Saving checkpoint
            print(' Epoch: {} model has been already save!'.format(epoch+1))

        #print the histogram
        #if epoch % 5 == 0:
        #    for name, param in backbone.named_parameters():
        #        if name in BACKBONE_PARAMS:
        #            log_writer.add_histogram(name + '_data', param.clone().cpu().data.numpy(), epoch)
        #            if param.grad is not None: #leaf node in the graph retain gradient
         #               log_writer.add_histogram(name + '_grad', param.grad, epoch)
        #param = sum(p.numel() for p in model.parameters() if p.requires_grad) #count params of model
        #print("\r Params of model: {}".format(count_bytes(param)) )
        #flops, _ = profile(model, inputs=(images,))
        #print("FLOPs(Floating Point Operations) of model = {}".format(count_bytes(flops)) )

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))
    #log_writer.close() #shut up the tensorboard

def Test(data_loader_box_test):

    print('********************load model********************')
    #resnet = resnet18(pretrained=False, num_classes=NUM_CLASSES).cuda()
    #backbone = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4)
    backbone = densenet121(pretrained=False, num_classes=NUM_CLASSES).features.cuda()
    backbone.out_channels = 1024 #resnet18=512,  densenet121=1024
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=NUM_CLASSES, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler).cuda()
    
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model.eval() 

    log_writer = SummaryWriter('/data/tmpexec/tensorboard-log') #--port 10002, start tensorboard
    for name, param in backbone.named_parameters():
        #print(name,'---', param.size())
        if name in BACKBONE_PARAMS:
            log_writer.add_histogram('cxr_' + name + '_data', param.clone().cpu().data.numpy())
    log_writer.close() #shut up the tensorboard
    print('********************load model succeed!********************')

    print('******* begin testing!*********')
    mAP = {0: [], 1: [], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[]}
    with torch.autograd.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader_box_test):
            images = list(image.cuda() for image in images)
            #images = list((image*torch.randn(image.size())).cuda() for image in images)#add Gaussian noisy
            targets = [{k:v.squeeze(0).cuda() for k, v in t.items()} for t in targets]
            var_output = model(images)#forward
        
            for i in range(len(targets)):
                gt_box = targets[i]['boxes'].cpu().data
                pred_box = var_output[i]['boxes'].cpu().data
                gt_lbl = targets[i]['labels'].cpu().data
                pred_lbl = var_output[i]['labels'].cpu().data
                for m in range(gt_box.shape[0]):
                    iou_max = 0.0
                    for n in range(pred_box.shape[0]):
                        if gt_lbl[m] == pred_lbl[n]:
                            iou = compute_iou(gt_box[m], pred_box[n])
                            if iou_max < iou: iou_max =  iou
                    if iou_max > 0.4: #hit
                        mAP[0].append(1)
                        mAP[gt_lbl[m].item()].append(1)
                    else:
                        mAP[0].append(0)
                        mAP[gt_lbl[m].item()].append(0)

            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
    for i in range(NUM_CLASSES):
        print('The mAP of {} is {:.4f}'.format(CLASS_NAMES_Vin[i], np.mean(mAP[i])))

def CVTETest():
    print('********************load data********************')
    dataloader_test = get_dataloader_CVTE(batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print('********************load data succeed!********************')

    print('********************load model********************')
    resnet = resnet18(pretrained=False, num_classes=NUM_CLASSES).cuda()
    backbone = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4)
    #backbone = densenet121(pretrained=False, num_classes=NUM_CLASSES).features.cuda()
    backbone.out_channels = 512 #resnet18=512,  densenet121=1024
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128),),aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=NUM_CLASSES, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler).cuda()
    
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model.eval() 
    print('******************** load model succeed!********************')

    print('******* begin testing!*********')
    mAP = {0: [], 1: [], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[]}
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    with torch.autograd.no_grad():
        for batch_idx, (image, label) in enumerate(dataloader_test):
            var_image = torch.autograd.Variable(image).cuda()
            var_output = model(var_image)#forward

            for i in range(len(label)):
                gt_lbl = label[i].item()
                if gt_lbl > 0:
                    pred_lbl = var_output[i]['labels'].cpu().data
                    hit = 0
                    for j in range(pred_lbl.shape[0]):
                        if gt_lbl == pred_lbl[j]: 
                            hit = 1
                    mAP[gt_lbl].append(hit)
                    mAP[0].append(hit)

                    #AUROC
                    gt = torch.cat((gt, torch.FloatTensor([1])), 0)
                    scores = var_output[i]['scores'].cpu().data
                    if len(scores)>0:
                        ind = np.argmax(scores)
                        pred = torch.cat((pred, scores[ind].unsqueeze(0)), 0)
                    else:
                        pred = torch.cat((pred, torch.FloatTensor([0.5])), 0)
                else:
                    #AUROC
                    gt = torch.cat((gt, torch.FloatTensor([0])), 0)
                    scores = var_output[i]['scores'].cpu().data
                    if len(scores)>0:
                        ind = np.argmax(scores)
                        pred = torch.cat((pred, scores[ind].unsqueeze(0)), 0)
                    else:
                        pred = torch.cat((pred, torch.FloatTensor([0.5])), 0)
                             
            sys.stdout.write('\r testing process: = {}'.format(batch_idx+1))
            sys.stdout.flush()
   
    for i in range(NUM_CLASSES):
        print('The mAP of {} is {:.4f}'.format(CLASS_NAMES_Vin[i], np.mean(mAP[i])))
    gt_np = gt.numpy()
    pred_np = pred.numpy()
    AUROCs = roc_auc_score(gt_np, pred_np)
    print(('The AUROC of CVTE is {:.4f}'.format(AUROCs)))

def main():
    print('********************load data********************')
    data_loader_box_train,  data_loader_box_test= get_box_dataloader_VIN(batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    print('********************load data succeed!********************')
    #Train(data_loader_box_train)
    Test(data_loader_box_test)
    #CVTETest()

if __name__ == '__main__':
    main()