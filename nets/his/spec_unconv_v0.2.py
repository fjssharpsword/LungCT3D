# encoding: utf-8
"""
Uncertainty Convolution with Spectral Weights
Author: Jason.Fang
Update time: 13/07/2021
"""

import math
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

class SpecUnConv(nn.Module):
    def __init__(self, module, name='weight'):
        super(SpecUnConv, self).__init__()
        self.module = module
        self.name = name
        self._make_params()

    def _l2normalize(self, x, eps=1e-12):
        return x / (x.norm() + eps)

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = self._l2normalize(w.data.new(height).normal_(0, 1))
        self.module.register_buffer(self.name + "_u", u)

    def _update_weight(self):

        w = getattr(self.module, self.name)
        u = getattr(self.module, self.name + "_u")

        height = w.data.shape[0]
        for _ in range(1):
            v = self._l2normalize(torch.mv(torch.t(w.view(height,-1).data), u))
            u = self._l2normalize(torch.mv(w.view(height,-1).data, v))

        setattr(self.module, self.name + "_u", u)
        w.data = w.data / torch.dot(u, torch.mv(w.view(height,-1).data, v))

    def forward(self, *args):
        self._update_weight()
        return self.module.forward(*args)

if __name__ == "__main__":

    #for debug  
    x =  torch.rand(2, 3, 10, 10).cuda()
    #sconv = SpecUnConv(nn.Conv2d(3, 16, kernel_size=3, padding=(3 - 1) // 2, stride=1, bias=False)).cuda()
    sconv = SpecUnConv(nn.Conv2d(3, 16, kernel_size=7, stride=2, padding=3, bias=False)).cuda()
    out = sconv(x)
    print(out.shape)
