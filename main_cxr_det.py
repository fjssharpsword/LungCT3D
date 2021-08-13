# encoding: utf-8
"""
Training implementation of object detection for 2D chest x-ray
Author: Jason.Fang
Update time: 29/07/2021
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
import matplotlib.pyplot as plt
from PIL import Image
import matplotlib.patches as patches
import cv2
import seaborn as sns
#define by myself
from utils.common import compute_iou, count_bytes
from data_cxr2d.vincxr_coco import get_box_dataloader_VIN
from data_cxr2d.CVTECXR_Test import get_dataloader_CVTE
from nets.resnet import resnet18
from nets.densenet import densenet121

#config
os.environ['CUDA_VISIBLE_DEVICES'] = "7"
CLASS_NAMES_Vin = ['Average', 'Aortic enlargement', 'Atelectasis', 'Calcification','Cardiomegaly', 'Consolidation', 'ILD', 'Infiltration', \
        'Lung Opacity', 'Nodule/Mass', 'Other lesion', 'Pleural effusion', 'Pleural thickening', 'Pneumothorax', 'Pulmonary fibrosis']
BACKBONE_PARAMS = ['4.0.conv1.weight', '4.0.conv1.weight_v', '4.0.conv1.grouped.weight', '4.0.conv1.weight_orig', '4.0.conv1.weight_p', '4.0.conv1.weight_q',\
                   '5.0.conv1.weight', '5.0.conv1.weight_v', '5.0.conv1.grouped.weight', '5.0.conv1.weight_orig', '5.0.conv1.weight_p', '5.0.conv1.weight_q', \
                   '6.0.conv1.weight', '6.0.conv1.weight_v', '6.0.conv1.grouped.weight', '6.0.conv1.weight_orig', '6.0.conv1.weight_p', '6.0.conv1.weight_q', \
                   '7.0.conv1.weight', '7.0.conv1.weight_v', '7.0.conv1.grouped.weight', '7.0.conv1.weight_orig', '7.0.conv1.weight_p', '7.0.conv1.weight_q' ]
BATCH_SIZE = 8
MAX_EPOCHS = 20
NUM_CLASSES =  len(CLASS_NAMES_Vin)
CKPT_PATH = '/data/pycode/LungCT3D/ckpt/vincxr_resnet_conv.pkl'

def Train():
    print('********************load data********************')
    data_loader_train = get_box_dataloader_VIN(batch_size=BATCH_SIZE, shuffle=True, num_workers=1)
    print('********************load data succeed!********************')

    print('********************load model********************')
    resnet = resnet18(pretrained=False, num_classes=NUM_CLASSES).cuda()
    backbone = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4)
    #backbone = densenet121(pretrained=False, num_classes=NUM_CLASSES).features.cuda()
    backbone.out_channels = 512 #resnet18=512,  densenet121=1024
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),aspect_ratios=((0.5, 1.0, 2.0),))
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
    log_writer = SummaryWriter('/data/tmpexec/tensorboard-log') #--port 10002, start tensorboard
    loss_min = float('inf')
    for epoch in range(MAX_EPOCHS):
        since = time.time()
        print('Epoch {}/{}'.format(epoch+1 , MAX_EPOCHS))
        print('-' * 10)
        model.train()  #set model to training mode
        train_loss = []
        with torch.autograd.enable_grad():
            for batch_idx, (images, targets) in enumerate(data_loader_train):
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
        if (epoch+1) % 4 == 0:
            for name, param in backbone.named_parameters():
                if name in BACKBONE_PARAMS:
                    log_writer.add_histogram(name + '_data', param.clone().cpu().data.numpy(), epoch)
                    if param.grad is not None: #leaf node in the graph retain gradient
                        log_writer.add_histogram(name + '_grad', param.grad, epoch)

        time_elapsed = time.time() - since
        print('Training epoch: {} completed in {:.0f}m {:.0f}s'.format(epoch+1, time_elapsed // 60 , time_elapsed % 60))
    log_writer.close() #shut up the tensorboard

def Test():
    print('********************load data********************')
    data_loader_test = get_box_dataloader_VIN(batch_size=BATCH_SIZE, shuffle=False, num_workers=1)
    print('********************load data succeed!********************')

    print('********************load model********************')
    resnet = resnet18(pretrained=False, num_classes=NUM_CLASSES).cuda()
    backbone = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4)
    #backbone = densenet121(pretrained=False, num_classes=NUM_CLASSES).features.cuda()
    backbone.out_channels = 512 #resnet18=512,  densenet121=1024
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=NUM_CLASSES, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler).cuda()

    for name, param in backbone.named_parameters():
        print(name,'---', param.size())
    
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model.eval() 

    """
    #plot the distribution of weights
    log_writer = SummaryWriter('/data/tmpexec/tensorboard-log') #--port 10002, start tensorboard
    epoch = 1
    for name, param in backbone.named_parameters():
        #print(name,'---', param.size())
        if name in ['4.0.conv1.weight_orig','5.0.conv1.weight_orig','6.0.conv1.weight_orig','7.0.conv1.weight_orig']:
            log_writer.add_histogram('sconv', param.clone().cpu().data.numpy(), epoch)
            epoch = epoch + 1
    log_writer.close() #shut up the tensorboard
    """
    print('********************load model succeed!********************')

    print('******* begin testing!*********')
    mAP = {0: [], 1: [], 2:[], 3:[], 4:[], 5:[], 6:[], 7:[], 8:[], 9:[], 10:[], 11:[], 12:[], 13:[], 14:[]}
    with torch.autograd.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader_test):
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

def VisFeature():
    print('********************load data********************')
    data_loader_test = get_box_dataloader_VIN(batch_size=1, shuffle=True, num_workers=0)
    print('********************load data succeed!********************')

    print('********************load model********************')
    resnet = resnet18(pretrained=False, num_classes=NUM_CLASSES).cuda()
    backbone = nn.Sequential(resnet.conv1, resnet.bn1,resnet.relu, resnet.maxpool,resnet.layer1,resnet.layer2,resnet.layer3,resnet.layer4)
    #backbone = densenet121(pretrained=False, num_classes=NUM_CLASSES).features.cuda()
    backbone.out_channels = 512 #resnet18=512,  densenet121=1024
    anchor_generator = AnchorGenerator(sizes=((32, 64, 128, 256),),aspect_ratios=((0.5, 1.0, 2.0),))
    roi_pooler = torchvision.ops.MultiScaleRoIAlign(featmap_names=['0'],output_size=7,sampling_ratio=2)
    model = FasterRCNN(backbone, num_classes=NUM_CLASSES, rpn_anchor_generator=anchor_generator, box_roi_pool=roi_pooler).cuda()

    #for name, param in backbone.named_parameters():
    #    print(name,'---', param.size())
    
    if os.path.exists(CKPT_PATH):
        checkpoint = torch.load(CKPT_PATH)
        model.load_state_dict(checkpoint) #strict=False
        print("=> Loaded well-trained checkpoint from: " + CKPT_PATH)
    model.eval() 
    print('******************** load model succeed!********************')

    print('*******Plot!*********')
    #log_writer = SummaryWriter('/data/tmpexec/tensorboard-log') #--port 10002, start tensorboard
    with torch.autograd.no_grad():
        for batch_idx, (images, targets) in enumerate(data_loader_test):
            images = list(image.cuda() for image in images)
            targets = [{k:v.squeeze(0).cuda() for k, v in t.items()} for t in targets]
            fea_map = model.backbone(images[0].unsqueeze(0))#forward
            #log_writer.add_histogram('cxr_fea', fea_map, 2)
            plt.hist(fea_map.cpu().numpy().flatten(), facecolor='r', alpha=0.75, density=True, linewidth=0.5) #orange
            #ax = sns.distplot(fea_map.cpu().numpy().flatten())
            plt.savefig('/data/pycode/LungCT3D/imgs/fea_5.jpg')

            img = images[0]
            box = targets[0]['boxes'][0]
            lbl = targets[0]['labels'][0]
            #plot goundtruth box
            fig, ax = plt.subplots()# Create figure and axes
            img = img.cpu().numpy().transpose(1,2,0)
            ax.imshow(img)
            rect = patches.Rectangle((box[0], box[1]), box[2]-box[0], box[3]-box[1], linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)# add groundtruth
            ax.text(box[0], box[1], CLASS_NAMES_Vin[lbl])
            ax.axis('off')
            fig.savefig('/data/pycode/LungCT3D/imgs/img_3.jpg')
            #log_writer.add_histogram('cxr_data', img, 2)
          
            """
            fea_map = fea_map.cpu().numpy().squeeze()
            fea_map = np.mean(fea_map, axis=0)
            #fea_map = np.maximum(fea_map, 0)
            #fea_map /= np.max(fea_map)
            #plt.matshow(fea_map)
            #plt.savefig('/data/pycode/LungCT3D/imgs/fea_1.jpg')
            fea_map = cv2.resize(fea_map, (img.shape[0], img.shape[1]))  
            fea_map = np.uint8(255 * fea_map) 
            fea_map = cv2.applyColorMap(fea_map, cv2.COLORMAP_JET)  
            img = np.uint8(255 * img) 
            overlay_img = fea_map * 0.3 + img 
            cv2.imwrite('/data/pycode/LungCT3D/imgs/fea_1.jpg', overlay_img)
            """
            break
    #log_writer.close() #shut up the tensorboard

def main():
    #Train()
    #Test()
    VisFeature()

if __name__ == '__main__':
    main()