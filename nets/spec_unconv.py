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
    def __init__(self, module, name='weight', sample=True):
        super(SpecUnConv, self).__init__()

        self.module = module
        self.name = name
        self.sample = sample
        #prior weight: mean and variance
        self.w_mu = nn.Parameter(torch.empty((self.module.out_channels, self.module.in_channels, *self.module.kernel_size)))
        self.w_sigma = nn.Parameter(torch.empty((self.module.out_channels, self.module.in_channels, *self.module.kernel_size)))
        self._make_params()

    def _make_params(self):
        #mean and variace 
        n = self.module.in_channels
        n *= self.module.kernel_size[0] ** 2
        stdv = 1.0 / math.sqrt(n)
        self.w_mu.data.normal_(-stdv, stdv)
        self.w_sigma.data.fill_(stdv)

        self.module.register_parameter(self.name + "_mu", self.w_mu)
        self.module.register_parameter(self.name + "_sigma", self.w_sigma)

    def _update_w(self):

        if self.sample: #sampling, distribution of weights learning
            w_eps = torch.empty(self.w_mu.size()).normal_(0, 1).cuda()
            w_sigma = torch.log1p(torch.exp(self.w_sigma))
            weight = self.w_mu + w_eps * w_sigma
            del self.module._parameters[self.name]
        else: #fixed value learning
            weight = getattr(self.module, self.name)

        u, s, v = torch.svd(weight.view(weight.size(0), -1))
        sigma =  torch.max(s) #maximum singular value

        setattr(self.module, self.name, weight / sigma.expand_as(weight))

    def forward(self, *args):
        self._update_w()
        return self.module.forward(*args)

if __name__ == "__main__":

    #for debug  
    x =  torch.rand(2, 3, 10, 10).cuda()
    sconv = SpecUnConv(nn.Conv2d(3, 16, kernel_size=3, padding=(3 - 1) // 2, stride=1, bias=False)).cuda()
    out = sconv(x)
    print(out.shape)
