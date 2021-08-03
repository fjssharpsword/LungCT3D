# encoding: utf-8
"""
Spectral Convolution based on Matrix Factorization
Author: Jason.Fang
Update time: 30/07/2021
"""

import math
import torch
import torch.nn as nn
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules import conv

class SpecConv2d(conv._ConvNd):
    r"""
    Applies Spectral Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, mf_k=10):
        padding = (kernel_size - 1) // 2
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(1)
        groups = 1
        bias = False
        super(SpecConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, 'zeros')

        #set projected matrices
        self._make_params(mf_k)

    def _make_params(self, mf_k):
        #spectral weight
        height = self.weight.shape[0]
        width = self.weight.view(height, -1).shape[1]

        p = nn.Parameter(torch.empty(height, mf_k), requires_grad=True)
        q = nn.Parameter(torch.empty(mf_k, width), requires_grad=True)

        nn.init.kaiming_normal_(p.data, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(q.data, mode='fan_out', nonlinearity='relu')

        self.register_parameter("weight_p", p)
        self.register_parameter("weight_q", q)

    @property
    def W_(self):
        #solve spectral norm
        _, s_p, v_p = torch.svd(self.weight_p.cpu()) #the speed in cpu is faster than in gpu
        u_q, s_q, _ = torch.svd(self.weight_q.cpu())
        #sigma = torch.max(s_p * torch.diag(v_p*u_q) * s_q).cuda()
        sigma = torch.sum(torch.abs(s_p * torch.diag(v_p*u_q) * s_q)).cuda()
        #approximate the weight
        w_hat = torch.mm(self.weight_p, self.weight_q).view_as(self.weight)

        del self.weight
        self.weight =  w_hat/sigma
        return self.weight

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride, self.padding, self.dilation, self.groups)

if __name__ == "__main__":
    #for debug  
    x =  torch.rand(2, 3, 16, 16).cuda()
    sconv = SpecConv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2).cuda()
    out = sconv(x)
    print(out.shape)

