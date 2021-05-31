# encoding: utf-8
"""
3D UNet
Author: Jason.Fang
Update time: 12/05/2021
"""
import re
import numpy as np
import math
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

# Generalized-Mean (GeM) pooling layer
# https://arxiv.org/pdf/1711.02512.pdf 
class GeMLayer(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeMLayer, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def _gem(self, x, p=3, eps=1e-6):
        return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(-3), x.size(-2), x.size(-1))).pow(1. / p)
        #return F.avg_pool2d(x.clamp(min=eps).pow(p), (x.size(-2), x.size(-1))).pow(1. / p)

    def forward(self, x):
        return self._gem(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

def positionalencoding1d(d_model, length):
        """
        #https://github.com/wzlxjtu/PositionalEncoding2D
        :param d_model: dimension of the model
        :param length: length of positions
        :return: length*d_model position matrix
        """
        if d_model % 2 != 0:
            raise ValueError("Cannot use sin/cos positional encoding with "
                            "odd dim (got dim={:d})".format(d_model))
        pe = torch.zeros(length, d_model)
        position = torch.arange(0, length).unsqueeze(1)
        div_term = torch.exp((torch.arange(0, d_model, 2, dtype=torch.float) * -(math.log(10000.0) / d_model)))
        pe[:, 0::2] = torch.sin(position.float() * div_term)
        pe[:, 1::2] = torch.cos(position.float() * div_term)

        return pe

class CT3DIRNet(nn.Module):
    def __init__(self, k_size = 5, code_size= 128):
        super(CT3DIRNet, self).__init__()

        self.linear_pro = nn.Sequential(
                          nn.Conv1d(1, 1, kernel_size=k_size, stride=(k_size - 1) // 2, padding=(k_size - 1) // 2, bias=False),
                          nn.Conv1d(1, 1, kernel_size=k_size, stride=(k_size - 1) // 2, padding=(k_size - 1) // 2, bias=False),
                          nn.Conv1d(1, 1, kernel_size=k_size, stride=(k_size - 1) // 2, padding=(k_size - 1) // 2, bias=False),
                          nn.Conv1d(1, 1, kernel_size=k_size, stride=(k_size - 1) // 2, padding=(k_size - 1) // 2, bias=False)
                          )
        self.bn = nn.BatchNorm1d(256)
        self.pos_embedding = nn.Parameter(torch.randn(1, 32, 256))
        self.trans_enc = nn.TransformerEncoderLayer(d_model=256*2, nhead=8)
        self.fc = nn.Sequential(nn.Linear(32*32, code_size), nn.Sigmoid()) #for metricl learning

    def forward(self, x):
        x = x.permute(0, 2, 1, 3, 4)
        B, D, C, H, W = x.shape
        x = x.view(B*D, C, H*W)
        x = self.linear_pro(x)
        x = x.squeeze()
        x = self.bn(x)
        x = x.view(B, D, -1)
        pos = self.pos_embedding.expand_as(x) 
        x = torch.cat((x, pos), dim=2)
        x = self.trans_enc(x)
        B, D, HW = x.shape
        x = x.view(B*D, 1, HW)
        x = self.linear_pro(x)
        x = x.squeeze()
        x = x.view(B, D, -1)
        x = x.view(B, -1)
        x = self.fc(x)

        return x



if __name__ == "__main__":
    #for debug  
    scan =  torch.rand(10, 1, 32, 64, 64)#.cuda()
    model = CT3DIRNet(k_size = 5, code_size = 128)#.cuda()
    out = model(scan)
    print(out.shape)

    