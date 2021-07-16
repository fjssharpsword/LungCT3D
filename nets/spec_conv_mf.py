# encoding: utf-8
"""
Spectral Convolution with Matrix Factorization
Author: Jason.Fang
Update time: 15/07/2021
"""

import math
import time
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

class SpecConv(nn.Module):
    def __init__(self, module, name='weight', mf_k = 10):
        super(SpecConv, self).__init__()
        self.module = module
        self.name = name
        self.mf_k = mf_k #latent factors
        self._make_params()

    def _l2normalize(self, x, eps=1e-12):
        return x / (x.norm() + eps)

    def _make_params(self):
        w = getattr(self.module, self.name)

        #spectral weight
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = nn.Parameter(torch.empty(height, self.mf_k), requires_grad=True)
        v = nn.Parameter(torch.empty(self.mf_k, width), requires_grad=True)
        s = nn.Parameter(torch.eye(self.mf_k), requires_grad=True)
        u.data = self._l2normalize(u.data.normal_(0, 1))
        v.data = self._l2normalize(v.data.normal_(0, 1))

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_s", s)

    def _update_weight(self):
        
        w = getattr(self.module, self.name)
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        s = getattr(self.module, self.name + "_s")

        w = torch.mm(torch.mm(u,s),v)
        sigma = torch.mean(torch.diag(s))

        w.data = w / sigma.expand_as(w) #rewrite weights

    def forward(self, *args):
        self._update_weight()
        return self.module.forward(*args)

if __name__ == "__main__":

    #for debug  
    x =  torch.rand(2, 3, 10, 10).cuda()
    #sconv = SpecUnConv(nn.Conv2d(3, 16, kernel_size=3, padding=(3 - 1) // 2, stride=1, bias=False)).cuda()
    sconv = SpecConv(nn.Conv2d(3, 4096, kernel_size=9, stride=2, padding=4, bias=False)).cuda()
    out = sconv(x)
    print(out.shape)
