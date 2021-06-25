# encoding: utf-8
"""
2D Retrieval Model for ImageNet
Author: Jason.Fang
Update time: 26/06/2021
"""
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#define by myself
#from nets.cba_2d import ChannelBayesianAttention
#from nets.ssa import SpatialSpectralAttention
from cba_2d import ChannelBayesianAttention
from ssa_2d import SpatialSpectralAttention

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.cba = ChannelBayesianAttention(k_size=k_size) #channel attention
        self.ssa = SpatialSpectralAttention(in_ch=planes, k=2, k_size=k_size) #spatial attention

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.cba(out)#channel attention
        out = self.ssa(out)#spatial attention

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)

        self.cba = ChannelBayesianAttention(k_size=k_size) #channel attention
        self.ssa = SpatialSpectralAttention(in_ch=planes * 4, k=2, k_size=k_size) #spatial attention

        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.cba(out)#channel attention
        out = self.ssa(out)#spatial attention

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, k_size=[3, 3, 3, 3]):
        self.inplanes = 64
        super(ResNet, self).__init__()
        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0], int(k_size[0]))
        self.layer2 = self._make_layer(block, 128, layers[1], int(k_size[1]), stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], int(k_size[2]), stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], int(k_size[3]), stride=2)
        self.avgpool = nn.AvgPool2d(7, stride=1)
        self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, k_size, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, k_size))
        self.inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.inplanes, planes, k_size=k_size))

        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)

        return x


def cba_ssa_resnet18(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def cba_ssa_resnet34(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def cba_ssa_resnet50(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def cba_ssa_resnet101(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def cba_ssa_resnet152(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ResNet(Bottleneck, [3, 8, 36, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

if __name__ == "__main__":
    #for debug  
    x =  torch.rand(2, 3, 256, 256).cuda()
    model = cba_ssa_resnet152(num_classes=64, k_size=[3, 3, 3, 3]).cuda()
    out = model(x)
    print(out.shape)