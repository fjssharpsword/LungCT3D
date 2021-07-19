# encoding: utf-8
"""
Spectral Convolution for 3D UNet.
Author: Jason.Fang
Update time: 17/07/2021
"""
import sys
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision
#define by myself
from nets.spec_conv_mf import SpecConv

def normalization(planes, norm='gn'):
    if norm == 'bn':
        m = nn.BatchNorm3d(planes)
    elif norm == 'gn':
        m = nn.GroupNorm(4, planes)
    elif norm == 'in':
        m = nn.InstanceNorm3d(planes)
    else:
        raise ValueError('normalization type {} is not supported'.format(norm))
    return m

class ConvD(nn.Module):
    def __init__(self, inplanes, planes, dropout=0.0, norm='gn', first=False):
        super(ConvD, self).__init__()

        self.first = first
        self.maxpool = nn.MaxPool3d(2, 2)

        self.dropout = dropout
        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        #self.conv1 = SpecConv(nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False))
        self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        #self.conv2 = SpecConv(nn.Conv3d(planes, planes, 3, 1, 1, bias=False))
        self.bn2   = normalization(planes, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        #self.conv3 = SpecConv(nn.Conv3d(planes, planes, 3, 1, 1, bias=False))
        self.bn3   = normalization(planes, norm)

    def forward(self, x):
        if not self.first:
            x = self.maxpool(x)
        x = self.bn1(self.conv1(x))
        y = self.relu(self.bn2(self.conv2(x)))
        if self.dropout > 0:
            y = F.dropout3d(y, self.dropout)
        y = self.bn3(self.conv3(y))
        return self.relu(x + y)

class ConvU(nn.Module):
    def __init__(self, planes, norm='gn', first=False):
        super(ConvU, self).__init__()

        self.first = first

        if not self.first:
            self.conv1 = nn.Conv3d(2*planes, planes, 3, 1, 1, bias=False)
            #self.conv1 = SpecConv(nn.Conv3d(2*planes, planes, 3, 1, 1, bias=False))
            self.bn1   = normalization(planes, norm)

        self.conv2 = nn.Conv3d(planes, planes//2, 1, 1, 0, bias=False)
        #self.conv2 = SpecConv(nn.Conv3d(planes, planes//2, 1, 1, 0, bias=False))
        self.bn2   = normalization(planes//2, norm)

        self.conv3 = nn.Conv3d(planes, planes, 3, 1, 1, bias=False)
        #self.conv3 = SpecConv(nn.Conv3d(planes, planes, 3, 1, 1, bias=False))
        self.bn3   = normalization(planes, norm)

        self.relu = nn.ReLU(inplace=True)

    def forward(self, x, prev):
        # final output is the localization layer
        if not self.first:
            x = self.relu(self.bn1(self.conv1(x)))

        y = F.upsample(x, scale_factor=2, mode='trilinear', align_corners=False)
        y = self.relu(self.bn2(self.conv2(y)))

        y = torch.cat([prev, y], 1)
        y = self.relu(self.bn3(self.conv3(y)))

        return y

class Conv3DMe(nn.Module):
    def __init__(self, inplanes, planes, norm='gn'):
        super(Conv3DMe, self).__init__()

        self.relu = nn.ReLU(inplace=True)

        self.conv1 = nn.Conv3d(inplanes, planes, 3, 1, 1, bias=False)
        self.bn1   = normalization(planes, norm)

    def forward(self, x):
        x = self.bn1(self.conv1(x))
        y = self.relu(x)

        return y

class ResidualBlock(nn.Module):
    def __init__(self, inp, oup, stride=1):
        super(ResidualBlock, self).__init__()
        kernel = 3
        self.block = nn.Sequential(
            nn.Conv3d(inp, inp, kernel, stride, int((kernel-1)/2), groups=inp, bias=False),
            normalization(inp, 'gn'),
            nn.ReLU(inplace=True),

            nn.Conv3d(inp, oup, 1, 1, 0, bias=False),
            normalization(oup, 'gn'),
            nn.ReLU(inplace=True),

            nn.Conv3d(in_channels=oup, out_channels=oup, kernel_size=3, stride=1, padding=1, groups=oup, bias=False),
            normalization(oup, 'gn'),
            nn.ReLU(inplace=True),
            nn.Conv3d(in_channels=oup, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
            normalization(oup, 'gn'),
        )
        if inp == oup:
            self.residual = None
        else:
            self.residual = nn.Sequential(
                nn.Conv3d(in_channels=inp, out_channels=oup, kernel_size=1, stride=1, padding=0, bias=False),
                normalization(oup, 'gn'),
            )
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        residual = x

        out = self.block(x)
        if self.residual is not None:
            residual = self.residual(x)

        out += residual
        return out


class Unet_Lobe(nn.Module):
    def __init__(self, input_channels=1, n=16, dropout=0.5, norm='gn', num_classes=7): #16 4
        super(Unet_Lobe, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2,
                mode='trilinear', align_corners=True)

        self.convd1 = ConvD(input_channels, n, dropout, norm, first=True)
        self.convd2 = ConvD(n, 2*n, dropout, norm)
        self.convd3 = ConvD(2*n, 4*n, dropout, norm)
        self.convd4 = ConvD(4*n, 8*n, dropout, norm)
        self.convd5 = ConvD(8*n,16*n, dropout, norm)

        self.mid_8 = nn.Sequential(
            nn.Conv3d(256, 256, 1, 1, 0, bias=False),
            normalization(256, norm),
            nn.ReLU())
        self.mid_16 = nn.Sequential(
            nn.Conv3d(128, 128, 1, 1, 0, bias=False),
            normalization(128, norm),
            nn.ReLU())
        self.mid_32 = nn.Sequential(
            nn.Conv3d(64, 64, 1, 1, 0, bias=False),
            normalization(64, norm),
            nn.ReLU())
        self.mid_64 = nn.Sequential(
            nn.Conv3d(32, 32, 1, 1, 0, bias=False),
            normalization(32, norm),
            nn.ReLU())
        self.mid_128 = nn.Sequential(
            nn.Conv3d(16, 16, 1, 1, 0, bias=False),
            normalization(16, norm),
            nn.ReLU())

        self.convu4 = ResidualBlock(16*n, 8*n)
        self.convu3 = ResidualBlock(12*n, 6*n)
        self.convu2 = ResidualBlock(8*n, 3*n)
        self.convu1 = ResidualBlock(4*n, 2*n)

        self.conv4 = Conv3DMe(256, 128)

        self.seg3 = nn.Conv3d(6*n, num_classes, 1)
        self.seg2 = nn.Conv3d(3*n, num_classes, 1)
        self.seg1 = nn.Conv3d(2*n, num_classes, 1)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x1 = self.convd1(x)
        x2 = self.convd2(x1)
        x3 = self.convd3(x2)
        x4 = self.convd4(x3)
        x5 = self.convd5(x4)

        x1 = self.mid_128(x1) + x1
        x2 = self.mid_64(x2) + x2
        x3 = self.mid_32(x3) + x3
        x4 = self.mid_16(x4) + x4
        x5 = self.mid_8(x5) + x5

        y5 = self.convu4(x5)
        y5 = F.upsample(y5, scale_factor=2, mode='trilinear', align_corners=True)
        y4 = torch.cat((x4, y5), dim=1)
        y4 = self.conv4(y4)  # 128

        y4 = F.upsample(y4, scale_factor=2, mode='trilinear', align_corners=True)
        y3 = torch.cat((x3, y4), dim=1)
        y3 = self.convu3(y3)  # 96
        
        y3 = F.upsample(y3, scale_factor=2, mode='trilinear', align_corners=True)
        y2 = torch.cat((x2, y3), dim=1)
        y2 = self.convu2(y2)  # 64
        
        y2 = F.upsample(y2, scale_factor=2, mode='trilinear', align_corners=True)
        y1 = torch.cat((x1, y2), dim=1)
        y1 = self.convu1(y1)

        y3 = self.seg3(y3)
        y2 = self.seg2(y2) + self.upsample(y3)
        y1 = self.seg1(y1) + y2

        return y1


class UNet3D(nn.Module):

    def __init__(self, num_classes=1):
        super(UNet3D, self).__init__()
        self.net = Unet_Lobe(input_channels=1, num_classes=num_classes)

    def forward(self, image):
        fpn_out = self.net(image)
        return torch.sigmoid(fpn_out)

class DiceLoss(nn.Module):
    def __init__(self):
        super(DiceLoss, self).__init__()
    
    def	forward(self, input, target):
        N = target.size(0)
        smooth = 1.
        
        input_flat = input.view(N, -1)
        target_flat = target.view(N, -1)
        
        intersection = input_flat * target_flat
        
        loss = 2 * (intersection.sum(1) + smooth) / (input_flat.sum(1) + target_flat.sum(1) + smooth)
        loss = 1 - loss.sum() / N
        
        return loss

if __name__ == "__main__":
    #epochs = 100
    #for debug   
    img = torch.rand(2, 1, 80, 80, 80).cuda()
    unet = UNet3D().cuda()
    out = unet(img)
    print(out.size())