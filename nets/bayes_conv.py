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
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _l2normalize(self, v, eps=1e-12):
        return v / (v.norm() + eps)

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = _l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = _l2normalize(torch.mv(w.view(height,-1).data, v.data))

        # sigma = torch.dot(u.data, torch.mv(w.view(height,-1).data, v.data))
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _make_params(self):
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)


    def forward(self, *args):
        self._update_u_v()
        return self.module.forward(*args)

class BayesConvNd(Module):
    r"""
    Applies Bayesian Convolution
    Arguments:
        convN: 1D, 2D, 3D.
        priors: {'prior_mu': 0, 'prior_sigma': 0.1}
                prior_mu (Float): mean of prior normal distribution.
                prior_sigma (Float): sigma of prior normal distribution.
    """
    def __init__(self, in_channels, out_channels, kernel_size, convN=2, stride=1, dilation=1, bias=True, priors=None):
        super(BayesConvNd, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.stride = stride
        self.padding = (kernel_size - 1) // 2 
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias

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

        conv_module = None

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
    bconv = BayesConvNd(in_channels=3, out_channels=16, kernel_size=5, convN=2, stride=1).cuda()
    out = bconv(x, sample=True)
    print(out.shape)