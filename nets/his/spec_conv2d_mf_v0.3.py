# encoding: utf-8
"""
Spectral Convolution based on Matrix Factorization
Author: Jason.Fang
Update time: 02/08/2021
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
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, mf_k=2): #[2,10,18]
        padding = (kernel_size - 1) // 2
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(1)
        groups = 1
        bias = False
        super(SpecConv2d, self).__init__(in_channels, out_channels, kernel_size, stride, padding, dilation, False, _pair(0), groups, bias, 'zeros')

        #the right largest singular value of W of power iteration (PI)
        #self.register_buffer('u', torch.Tensor(1, out_channels).normal_())
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
        #approximate the weight
        w_hat = torch.mm(self.weight_p, self.weight_q)

        #MF: solve spectral norm
        _, s_p, v_p = torch.svd(self.weight_p.cpu()) #the speed in cpu is faster than in gpu
        u_q, s_q, _ = torch.svd(self.weight_q.cpu())
        #sigma = torch.max(s_p * torch.diag(v_p*u_q) * s_q).cuda()
        sigma = torch.sum(torch.abs(s_p * torch.diag(v_p*u_q) * s_q)).cuda()

        #PI: solve spectral norm
        #sigma, _u = self._power_iteration(w_hat, self.u)
        #self.u.copy_(_u)

        #rewrite the weight
        w_hat = w_hat.view_as(self.weight)
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

