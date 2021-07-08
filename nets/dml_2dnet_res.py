# encoding: utf-8
"""
Bayesian convolution with spectral weight for 2D Resnet.
Author: Jason.Fang
Update time: 07/07/2021
"""
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

#define by myself
from nets.bayes_conv import BayesConvNd
#from bayes_conv import BayesConvNd

#Channel Bayesian attention
class ChannelBayesianAttention(nn.Module):
    """ Constructs a CBA module.
        Args:k_size: kernel size
    """
    def __init__(self, k_size=3):
        super(ChannelBayesianAttention, self).__init__()
        self.avg_2dpool = nn.AdaptiveAvgPool2d(1)
        #self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False) 
        self.bayes_conv = BayesConvNd(in_channels=1, out_channels=1, kernel_size=k_size, convN=1, stride=1, power_iterations=10)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        #cross-channel
        y = self.avg_2dpool(x) 
        y = y.squeeze(-1).transpose(-1, -2) #(B, C, 1, 1) -> (B, C, 1)->(B, 1, C)

        #y = self.conv(y) #local uncertain-dependencies
        y = self.bayes_conv(y) 
        y = y.transpose(2, 1).unsqueeze(-1) #(B, 1, C)-> (B, C, 1)-> (B, C, 1, 1)

        y = self.sigmoid(y)
        x = x * y.expand_as(x)# Multi-scale information fusion

        return x

#spatial Bayesian Attention (CBAM)
class SpatialBayesianAttention(nn.Module):
    """ Constructs a SBA module.
        Args:k_size: kernel size
    """
    def __init__(self, k_size=3):
        super(SpatialBayesianAttention, self).__init__()
        #self.conv = nn.Conv2d(2, 1, k_size, stride=1, padding=(k_size - 1) // 2)
        self.bayes_conv = BayesConvNd(in_channels=2, out_channels=1, kernel_size=k_size, convN=2, stride=1, power_iterations=10)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        residual = x 

        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)

        #x = self.conv(x)  
        x = self.bayes_conv(x)
        x = self.sigmoid(x)

        x = x * residual

        return x

def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)
    #return BayesConvNd(in_channels=in_planes, out_channels=out_planes, kernel_size=3, convN=2, stride=stride, power_iterations=10)

class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None, k_size=3):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, 1)
        self.bn2 = nn.BatchNorm2d(planes)

        self.downsample = downsample
        self.stride = stride

        self.cba = ChannelBayesianAttention(k_size=3) #attention
        self.sba = SpatialBayesianAttention(k_size=3)

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        x = self.cba(x) #attention
        x = self.sba(x)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out

class Bayes_ResNet(nn.Module):

    def __init__(self, block, layers, num_classes=1000, k_size=[3, 3, 3, 3]):
        self.inplanes = 64
        super(Bayes_ResNet, self).__init__()
        self.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
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


def bayes_resnet18(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-18 model.
    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = Bayes_ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def bayes_resnet34(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-34 model.
    Args:
        k_size: Adaptive selection of kernel size
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        num_classes:The classes of classification
    """
    model = Bayes_ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def bayes_resnet50(k_size=[3, 3, 3, 3], num_classes=1000, pretrained=False):
    """Constructs a ResNet-50 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Bayes_ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def bayes_resnet101(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-101 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Bayes_ResNet(BasicBlock, [3, 4, 23, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model


def bayes_resnet152(k_size=[3, 3, 3, 3], num_classes=1_000, pretrained=False):
    """Constructs a ResNet-152 model.
    Args:
        k_size: Adaptive selection of kernel size
        num_classes:The classes of classification
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = Bayes_ResNet(BasicBlock, [3, 8, 36, 3], num_classes=num_classes, k_size=k_size)
    model.avgpool = nn.AdaptiveAvgPool2d(1)
    return model

def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

if __name__ == "__main__":
    #for debug  
    x =  torch.rand(2, 1, 128, 128).cuda()
    model = bayes_resnet18(num_classes=10).cuda()
    out = model(x)
    print(out.shape)
    print(count_parameters(model))