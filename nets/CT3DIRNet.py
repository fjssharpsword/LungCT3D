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

#3DConv Feature Map
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

class Conv3DNet(nn.Module):
    def __init__(self, in_channels, model_depth=4, pool_size=(1,2,2)):
        super(Conv3DNet, self).__init__()
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
                self.pooling = nn.MaxPool3d(kernel_size=pool_size, stride=(1,2,2), padding=0)
                self.module_dict["max_pooling_{}".format(depth)] = self.pooling

    def forward(self, x):
        for k, op in self.module_dict.items():
            if k.startswith("conv"):
                x = op(x)
            elif k.startswith("max_pooling"):
                x = op(x)
        return x

# Generalized-Mean (GeM) pooling layer
# https://arxiv.org/pdf/1711.02512.pdf 
class GeM3D(nn.Module):
    def __init__(self, p=3, eps=1e-6):
        super(GeM3D, self).__init__()
        self.p = Parameter(torch.ones(1) * p)
        self.eps = eps

    def _gem3d(self, x, p=3, eps=1e-6):
        return F.avg_pool3d(x.clamp(min=eps).pow(p), (x.size(-3), x.size(-2), x.size(-1))).pow(1. / p)

    def forward(self, x):
        return self._gem3d(x, p=self.p, eps=self.eps)

    def __repr__(self):
        return self.__class__.__name__ + '(' + 'p=' + '{:.4f}'.format(self.p.data.tolist()[0]) + ', ' + 'eps=' + str(self.eps) + ')'

#Cross-Slice Attention
class CrossSliceAttention(nn.Module):
    """Constructs a CSA module.
    Args:k_size: Adaptive selection of kernel size
    """
    def __init__(self, k_size=3):
        super(CrossSliceAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        B, C, D, H, W = x.shape
        x = x.view(B, C*D, H, W)
        y = self.avg_pool(x)
        # Two different branches of ECA module
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        x = x * y.expand_as(x)
        x = x.view(B, C, D, H, W)
        return x

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

class NonLocalAttention(nn.Module): #self-attention block
    def __init__(self, in_ch, k):
        super(NonLocalAttention, self).__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // k

        #print('Num channels:  in    out    mid')
        #print('               {:>4d}  {:>4d}  {:>4d}'.format(self.in_ch, self.out_ch, self.mid_ch))

        self.f = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU())
        self.g = nn.Sequential(
            nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1)),
            nn.BatchNorm2d(self.mid_ch),
            nn.ReLU())
        self.h = nn.Conv2d(self.in_ch, self.mid_ch, (1, 1), (1, 1))
        self.v = nn.Conv2d(self.mid_ch, self.out_ch, (1, 1), (1, 1))

        self.softmax = nn.Softmax(dim=-1)

        for conv in [self.f, self.g, self.h]: 
            conv.apply(weights_init)
        self.v.apply(constant_init)

    def forward(self, x):
        B, C, D, H, W = x.shape
        x = x.view(B, C*D, H, W)

        f_x = self.f(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W
        g_x = self.g(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W
        h_x = self.h(x).view(B, self.mid_ch, H * W)  # B * mid_ch * N, where N = H*W

        z = torch.bmm(f_x.permute(0, 2, 1), g_x)  # B * N * N, where N = H*W
        attn = self.softmax((self.mid_ch ** -.50) * z)

        z = torch.bmm(attn, h_x.permute(0, 2, 1))  # B * N * mid_ch, where N = H*W
        z = z.permute(0, 2, 1).view(B, self.mid_ch, H, W)  # B * mid_ch * H * W

        z = self.v(z)
        x = torch.add(z, x) # z + x

        x = x.view(B, C, D, H, W)

        return x

## Kaiming weight initialisation
def weights_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass
def constant_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.constant_(module.weight.data, 0.0)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass

class CT3DIRNet(nn.Module):
    def __init__(self, in_channels, code_size=512, model_depth=4, final_activation="sigmoid"):
        super(CT3DIRNet, self).__init__()
        self.conv3d = Conv3DNet(in_channels=in_channels, model_depth=model_depth)
        if final_activation == "sigmoid":
            self.sigmoid = nn.Sigmoid()
        else:
            self.softmax = nn.Softmax(dim=1)
        self.nla = NonLocalAttention(in_ch=512*8, k=2)
        self.csa = CrossSliceAttention()
        self.gem3d = GeM3D()
        self.fc = nn.Sequential(nn.Linear(512, code_size), nn.Sigmoid()) #for metricl learning

    def forward(self, x):
        x = self.conv3d(x)
        x = self.csa(x)
        x = self.nla(x)
        x = self.gem3d(x).view(x.size(0), -1)
        x = self.fc(x)
        return x

if __name__ == "__main__":
    #for debug  
    scan =  torch.rand(10, 1, 8, 256, 256)#.cuda()
    model = CT3DIRNet(in_channels=1, code_size=8)#.cuda()
    out = model(scan)
    print(out.shape)