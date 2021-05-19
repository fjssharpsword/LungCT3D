# encoding: utf-8
"""
3D UNet
Author: Jason.Fang
Update time: 12/05/2021
"""
import re
import numpy as np
import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F
from torch.nn.parameter import Parameter
import torch.utils.model_zoo as model_zoo
from collections import OrderedDict
from skimage.measure import label
import cv2
import torchvision.transforms as transforms
from torch.autograd import Variable
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image
#define by myself

#https://github.com/JielongZ/3D-UNet-PyTorch-Implementation/blob/master/unet3d_model/building_components.py
class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, k_size=3, stride=1, padding=1):
        super(ConvBlock, self).__init__()
        self.conv3d = nn.Conv3d(in_channels=in_channels, out_channels=out_channels, kernel_size=k_size,
                                stride=stride, padding=padding)
        self.batch_norm = nn.BatchNorm3d(num_features=out_channels)

    def forward(self, x):
        x = self.batch_norm(self.conv3d(x))
        x = F.elu(x)
        return x

class EncoderBlock(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=2):
        super(EncoderBlock, self).__init__()
        self.root_feat_maps = 16
        self.num_conv_blocks = 2
        # self.module_list = nn.ModuleList()
        self.module_dict = nn.ModuleDict()
        for depth in range(model_depth):
            feat_map_channels = 2 ** (depth + 1) * self.root_feat_maps
            for i in range(self.num_conv_blocks):
                # print("depth {}, conv {}".format(depth, i))
                if depth == 0:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
                else:
                    # print(in_channels, feat_map_channels)
                    self.conv_block = ConvBlock(in_channels=in_channels, out_channels=feat_map_channels)
                    self.module_dict["conv_{}_{}".format(depth, i)] = self.conv_block
                    in_channels, feat_map_channels = feat_map_channels, feat_map_channels * 2
            if depth == model_depth - 1:
                break
            else:
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=2, padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        down_sampling_features = []
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)
                #print(k, x.shape)
                if k.endswith("1"):
                    down_sampling_features.append(x)
            elif k.startswith("max_pooling"):
                x = op(x)
                #print(k, x.shape)

        return x, down_sampling_features

def gem3d(x, p=3, eps=1e-6):
    return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(-3), x.size(-2), x.size(-1))).pow(1. / p)

class GeM3D(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM3D, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def forward(self, x):
        return gem3d(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

class SpatialAttention(nn.Module):#spatial attention layer
    def __init__(self):
        super(SpatialAttention, self).__init__()

        self.conv1 = nn.Conv2d(2, 1, kernel_size=3, padding=1, bias=False)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        avg_out = torch.mean(x, dim=2, keepdim=True)
        max_out, _ = torch.max(x, dim=2, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=2)
        x = torch.squeeze(x, 1)
        x = self.conv1(x)
        x = torch.unsqueeze(x, 1)
        return self.sigmoid(x)

#https://github.com/qianjinhao/circle-loss/blob/master/circle_loss.py
class CircleLoss(nn.Module):
    def __init__(self, scale=32, margin=0.25, similarity='cos', **kwargs):
        super(CircleLoss, self).__init__()
        self.scale = scale
        self.margin = margin
        self.similarity = similarity

    def forward(self, feats, labels):
        assert feats.size(0) == labels.size(0), \
            f"feats.size(0): {feats.size(0)} is not equal to labels.size(0): {labels.size(0)}"
        
        labels = (labels.cpu().data + 1)
        labels = labels.unsqueeze(1)
        mask = torch.matmul(labels, torch.t(labels))
        mask = mask.eq(1).int() + mask.eq(4).int() + mask.eq(9).int() + mask.eq(16).int() + mask.eq(25).int()#1x1,2x2,3x3,4x4,5x5

        pos_mask = mask.triu(diagonal=1)
        neg_mask = (mask - 1).abs_().triu(diagonal=1)
        if self.similarity == 'dot':
            sim_mat = torch.matmul(feats, torch.t(feats))
        elif self.similarity == 'cos':
            feats = F.normalize(feats)
            sim_mat = feats.mm(feats.t())
        else:
            raise ValueError('This similarity is not implemented.')

        pos_pair_ = sim_mat[pos_mask == 1]
        neg_pair_ = sim_mat[neg_mask == 1]

        alpha_p = torch.relu(-pos_pair_ + 1 + self.margin)
        alpha_n = torch.relu(neg_pair_ + self.margin)
        margin_p = 1 - self.margin
        margin_n = self.margin
        loss_p = torch.sum(torch.exp(-self.scale * alpha_p * (pos_pair_ - margin_p)))
        loss_n = torch.sum(torch.exp(self.scale * alpha_n * (neg_pair_ - margin_n)))
        loss = torch.log(1 + loss_p * loss_n)
        return loss

class CT3DClassifier(nn.Module):
    def __init__(self, in_channels, num_classes, model_depth=4, final_activation="sigmoid"):
        super(CT3DClassifier, self).__init__()
        self.sa = SpatialAttention()
        self.encoder = EncoderBlock(in_channels=in_channels, model_depth=model_depth)
        if final_activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)
        self.gem3d = GeM3D()
        #self.classifier = nn.Sequential(nn.Linear(512, num_classes), nn.Softmax(dim=1))
        self.classifier = nn.Sequential(nn.Linear(512, num_classes), nn.Sigmoid())

    def forward(self, x):
        x = self.sa(x)*x 
        x, downsampling_features = self.encoder(x)
        x = self.gem3d(x).view(x.size(0), -1)
        feat = self.sigmoid(x)
        out = self.classifier(feat)
        return out, feat

if __name__ == "__main__":
    #for debug  
    """
    slice = torch.rand(16, 1, 32, 64, 64).cuda()
    mask = torch.rand(16, 1, 32, 64, 64).cuda()
    #encoder = EncoderBlock(in_channels=1).cuda()
    #enc_out, enc_h = encoder(slice)
    #decoder = DecoderBlock(out_channels=1).cuda()
    #dec_out = decoder(enc_out, enc_h)
    unet = CT3DUnetModel(in_channels=1, out_channels=1).cuda()
    out, feat = unet(slice)
    print(feat.shape)
    dl = DiceLoss().cuda()
    print(dl(mask, out).item())
    """

    scan =  torch.rand(16, 1, 8, 256, 256).cuda()
    model = CT3DClassifier(in_channels=1, num_classes=5).cuda()
    out, feat = model(scan)
    print(feat.shape)
    print(out.shape)

