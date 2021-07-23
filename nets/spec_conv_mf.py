# encoding: utf-8
"""
Spectral Convolution with Matrix Factorization
Author: Jason.Fang
Update time: 16/07/2021
"""

import math
import time
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

class SpecConv(nn.Module):
    def __init__(self, module, name='weight', mf_k = 10):#k =[1, 5, 10]
        super(SpecConv, self).__init__()
        self.module = module
        self.name = name
        self.mf_k = mf_k #latent factors
        self._make_params()

    def _l2normalize(self, x, eps=1e-12):
        return x / (x.norm() + eps)

    def _make_params(self):
        #spectral weight
        w = getattr(self.module, self.name)
        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        p = nn.Parameter(torch.empty(height, self.mf_k), requires_grad=True)
        q = nn.Parameter(torch.empty(self.mf_k, width), requires_grad=True)

        nn.init.kaiming_normal_(p.data, mode='fan_out', nonlinearity='relu')
        nn.init.kaiming_normal_(q.data, mode='fan_out', nonlinearity='relu')

        self.module.register_parameter(self.name + "_p", p)
        self.module.register_parameter(self.name + "_q", q)

    def _update_weight(self):
        #get parameters
        w = getattr(self.module, self.name)
        p = getattr(self.module, self.name + "_p")
        q = getattr(self.module, self.name + "_q")
        #solve simga
        #_, s_p, v_p = torch.svd(p.cpu()) #the speed in cpu is faster than in gpu
        #u_q, s_q, _ = torch.svd(q.cpu())
        #sigma = torch.max(s_p * torch.diag(v_p*u_q) * s_q).cuda()
        #rewrite weights
        w.data = torch.mm(p,q).view_as(w)
        #w.data = w / sigma.expand_as(w) 

    def forward(self, *args):
        self._update_weight()
        return self.module.forward(*args)

if __name__ == "__main__":

    #for debug  
    x =  torch.rand(2, 3, 32, 32).cuda()
    #sconv = SpecUnConv(nn.Conv2d(3, 16, kernel_size=3, padding=(3 - 1) // 2, stride=1, bias=False)).cuda()
    sconv = SpecConv(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False), mf_k = 10).cuda()
    out = sconv(x)
    print(out.shape)
