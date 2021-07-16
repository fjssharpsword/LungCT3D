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

        p = nn.Parameter(torch.empty(height, self.mf_k), requires_grad=True)
        q = nn.Parameter(torch.empty(self.mf_k, width), requires_grad=True)
        p.data.normal_(0, 1)
        q.data.normal_(0, 1)

        self.module.register_parameter(self.name + "_p", p)
        self.module.register_parameter(self.name + "_q", q)

    def _update_weight(self):
        
        w = getattr(self.module, self.name)
        p = getattr(self.module, self.name + "_p")
        q = getattr(self.module, self.name + "_q")

        p = self._l2normalize(p)
        q = self._l2normalize(q)
        _, p_s, _ = torch.svd(p.cpu()) #the speed in cpu is faster than in gpu
        _, q_s, _ = torch.svd(q.cpu())
        sigma = torch.max(torch.max(p_s),torch.max(q_s)).cuda()

        w = torch.mm(p,q).view_as(w)
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
