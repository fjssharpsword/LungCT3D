# encoding: utf-8
"""
Weight Normalization.
Author: Jason.Fang
Update time: 26/07/2021
"""

import math
import time
import torch
import torch.nn.init as init
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

class WeightNormalization(nn.Module):
    def __init__(self, module, name='weight'):
        super(WeightNormalization, self).__init__()
        self.module = module
        self.name = name

    def _l2normalize(self, x, eps=1e-12):
        return x / (x.norm() + eps)

    def _minmaxscaler(self, x):
        min = x.min()
        max = x.max()
        return (x - min)/(max-min)

    def _update_weight(self):
        #get parameters
        w = getattr(self.module, self.name)
        #w.data = self._l2normalize(w)
        w.data = self._minmaxscaler(w)

    def forward(self, *args):
        self._update_weight()
        return self.module.forward(*args)

if __name__ == "__main__":

    #for debug  
    x =  torch.rand(2, 3, 32, 32).cuda()
    wnorm = WeightNormalization(nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)).cuda()
    out = wnorm(x)
    print(out.shape)
