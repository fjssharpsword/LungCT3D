# encoding: utf-8
"""
Uncertainty Convolution with Spectral Weights
Author: Jason.Fang
Update time: 14/07/2021
"""

import math
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

class SpecUnConvNd(nn.Module):
    r"""
    Applies Bayesian Convolution
    Arguments:
        convN: 1D, 2D, 3D.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, convN=2):
        super(SpecUnConvNd, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = (kernel_size - 1) // 2 
        self.bias = None
        self.dilation = 1
        self.groups = 1
        self.convN = convN
        if self.convN == 1: #1D
            self.kernel_size = _single(kernel_size)
        elif self.convN ==2: #2D
            self.kernel_size = _pair(kernel_size)
        elif self.convN ==3: #3D
            self.kernel_size = _triple(kernel_size)
        else:
             raise ValueError('ConvN must be 1,2,3')

        #prior weight: mean and variance
        self.w_mu = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
        self.w_sigma = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
        self._make_params()

    def _make_params(self):
        #mean and variace 
        n = self.in_channels
        n *= self.kernel_size[0] ** 2
        stdv = 1.0 / math.sqrt(n)
        self.w_mu.data.normal_(-stdv, stdv)
        self.w_sigma.data.fill_(stdv)

    def forward(self, input, sample=True):

        if sample: #sampling, distribution of weights learning
            w_eps = torch.empty(self.w_mu.size()).normal_(0, 1).cuda()
            w_sigma = torch.log1p(torch.exp(self.w_sigma))
            weight = self.w_mu + w_eps * w_sigma
        else: #fixed value learning
            weight = self.w_mu
        #spectral- maximum singular value
        u, s, v = torch.svd(weight.view(weight.size(0), -1))
        sigma =  torch.max(s) #maximum singular value
        weight = weight / sigma.expand_as(weight)
         
        if self.convN == 1: #1D
            conv_module =  F.conv1d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.convN ==2: #2D
            conv_module = F.conv2d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.convN ==3: #3D
            conv_module = F.conv3d(input, weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        else: conv_module = None
        
        return conv_module

if __name__ == "__main__":
    #for debug  
    x =  torch.rand(2, 3, 80, 80).cuda()
    sconv = SpecUnConvNd(in_channels=3, out_channels=16, kernel_size=3, stride=1, convN=2).cuda()
    out = sconv(x, sample=True)
    print(out.shape)

    #specnm = SpectralNorm(power_iterations=100).cuda()
    #out = specnm(out)
    #print(out.shape)
