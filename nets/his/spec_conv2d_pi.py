# encoding: utf-8
"""
Spectral Convolution based on Power Iteration
Author: Jason.Fang
Update time: 30/07/2021
"""

import math
import torch
import torch.nn.init as init
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple
from torch.nn.modules import conv

#https://github.com/godisboy/SN-GAN/blob/master/src/snlayers/snconv2d.py
class SpecConv2d(conv._ConvNd):
    r"""
    Applies Spectral Convolution
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1):
        padding = (kernel_size - 1) // 2
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(1)
        groups = 1
        bias = False
        super(SpecConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, 'zeros')

        #the right largest singular value of W of power iteration (PI)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    def _l2normalize(self, v, eps=1e-12):
        return v / (torch.norm(v) + eps)
    def _power_iteration(self, W, u=None, Ip=1):
        """
        power iteration for max_singular_value
        """
        #xp = W.data
        if not Ip >= 1:
            raise ValueError("Power iteration should be a positive integer")
        if u is None:
            u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
        _u = u
        for _ in range(Ip):
            _v = self._l2normalize(torch.matmul(_u, W.data), eps=1e-12)
            _u = self._l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
        sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
        return sigma, _u

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = self._power_iteration(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):

        return F.conv2d(input, self.W_, self.bias, self.stride, self.padding, self.dilation, self.groups)

if __name__ == "__main__":
    #for debug  
    x =  torch.rand(2, 3, 16, 16).cuda()
    sconv = SpecConv2d(in_channels=3, out_channels=16, kernel_size=5, stride=2).cuda()
    out = sconv(x)
    print(out.shape)

