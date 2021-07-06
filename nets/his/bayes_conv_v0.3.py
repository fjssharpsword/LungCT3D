# encoding: utf-8
"""
Bayesian Convolution
Author: Jason.Fang
Update time: 02/07/2021
"""

import math
import torch
import torch.nn.init as init
from torch.nn import Module, Parameter
import torch.nn.functional as F
from torch.nn.modules.utils import _single, _pair, _triple

#weights with spectral normalization
class SpectralNorm(Module):
    def __init__(self, power_iterations=1):
        super(SpectralNorm, self).__init__()

        self.power_iterations = power_iterations

    def _l2normalize(self, x, eps=1e-12):
        return x / (x.norm() + eps)

    def forward(self, x):

        mu =  torch.mean(x).data
        var = torch.var(x).data

        height = x.data.shape[0]
        width = x.view(height, -1).data.shape[1]

        u = Parameter(x.data.new(height).normal_(mu, var), requires_grad=False) #mu=0,var=1
        v = Parameter(x.data.new(width).normal_(mu, var), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)

        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.mv(torch.t(x.view(height,-1).data), u.data))
            u.data = self._l2normalize(torch.mv(x.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(x.view(height, -1).mv(v))
        x = x / sigma.expand_as(x)

        return x

class BayesConvNd(Module):
    r"""
    Applies Bayesian Convolution
    Arguments:
        convN: 1D, 2D, 3D.
        priors: {'prior_mu': 0, 'prior_sigma': 0.5}
                prior_mu (Float): mean of prior normal distribution.
                prior_sigma (Float): sigma of prior normal distribution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, convN=2, stride=1, power_iterations=10, dilation=1, bias=True, priors=None):
        super(BayesConvNd, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = (kernel_size - 1) // 2 
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias
        self.power_iterations = power_iterations

        self.convN = convN
        if self.convN == 1: #1D
            self.kernel_size = _single(kernel_size)
        elif self.convN ==2: #2D
            self.kernel_size = _pair(kernel_size)
        elif self.convN ==3: #3D
            self.kernel_size = _triple(kernel_size)
        else:
             raise ValueError('ConvN must be 1,2,3')

        #prior 
        if priors is None:
            priors = {'prior_mu': 0, 'prior_sigma': 0.5}
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.prior_log_sigma = math.log(self.prior_sigma)

        #posterior weight: mean and variance
        self.W_mu = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
        self.W_rho = Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
        #bias
        if self.use_bias:
            self.bias_mu = Parameter(torch.empty((out_channels)))
            self.bias_rho = Parameter(torch.empty((out_channels)))

        # Initializating posterior
        self._reset_parameters()

    def _reset_parameters(self):
        # Initialization method of Adv-BNN.
        n = self.in_channels
        n *= self.kernel_size[0] ** 2
        stdv = 1.0 / math.sqrt(n)

        self.W_mu.data.normal_(-stdv, stdv)
        self.W_rho.data.fill_(self.prior_log_sigma)

        if self.use_bias:
            self.bias_mu.data.uniform_(-stdv, stdv)
            self.bias_rho.data.fill_(self.prior_log_sigma)

    def _l2normalize(self, x, eps=1e-12):
        return x / (x.norm() + eps)

    def _spec_norm(self, w):

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(self.prior_mu, self.prior_sigma), requires_grad=False) #mu=0,var=1
        v = Parameter(w.data.new(width).normal_(self.prior_mu, self.prior_sigma), requires_grad=False)
        u.data = self._l2normalize(u.data)
        v.data = self._l2normalize(v.data)

        for _ in range(self.power_iterations):
            v.data = self._l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = self._l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        w = w / sigma.expand_as(w)

        return w

    def forward(self, input, sample=True):

        if sample: #sampling, Bayesian Convolution
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).cuda()
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).cuda()
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None

        else: #no sampling, Convolution
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        weight = self._spec_norm(weight) #spectral weight
         
        if self.convN == 1: #1D
            conv_module =  F.conv1d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.convN ==2: #2D
            conv_module = F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        elif self.convN ==3: #3D
            conv_module = F.conv3d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        else: conv_module = None
        
        return conv_module

if __name__ == "__main__":
    #for debug  
    x =  torch.rand(2, 3, 80, 80).cuda()
    bconv = BayesConvNd(in_channels=3, out_channels=16, kernel_size=5, convN=2, stride=1, power_iterations=10).cuda()
    out = bconv(x, sample=True)
    print(out.shape)

    #specnm = SpectralNorm(power_iterations=100).cuda()
    #out = specnm(out)
    #print(out.shape)
