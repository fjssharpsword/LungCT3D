# encoding: utf-8
"""
Uncertainty Channel Attention
Author: Jason.Fang
Update time: 18/06/2021
"""
import sys
import math
import torch
import torch.nn as nn
import torch.nn.functional as F

class BayesConv1d(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=1, dilation=1, bias=True, priors=None):
        super(BayesConv1d, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = (kernel_size,) #if isinstance(kernel_size, tuple) else (kernel_size, kernel_size) #for conv2
        self.stride = stride
        self.padding = padding #(kernel_size - 1) // 2 
        self.dilation = dilation
        self.groups = 1
        self.use_bias = bias

        #prior 
        if priors is None:
            priors = {'prior_mu': 0, 'prior_sigma': 0.1}
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.prior_log_sigma = math.log(self.prior_sigma)

        #posterior weight: mean and variance
        self.W_mu = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
        self.W_rho = nn.Parameter(torch.empty((out_channels, in_channels, *self.kernel_size)))
        #bias
        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty((out_channels)))
            self.bias_rho = nn.Parameter(torch.empty((out_channels)))

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
        if sample: #training
            W_eps = torch.empty(self.W_mu.size()).normal_(0, 1).cuda()
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = torch.empty(self.bias_mu.size()).normal_(0, 1).cuda()
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else: #testing
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.conv1d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)
        #return F.conv2d(input, weight, bias, self.stride, self.padding, self.dilation, self.groups)

    #KL loss
    def _calculate_kl(self, mu_q, sig_q, mu_p, sig_p):
        kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
        return kl

    def kl_loss(self):
        kl = self._calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += self._calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl

class BayesFC(nn.Module):
    def __init__(self, in_features, out_features, bias=True, priors=None):
        super(BayesFC, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.use_bias = bias

        if priors is None:
            priors = {'prior_mu': 0, 'prior_sigma': 0.1}
        self.prior_mu = priors['prior_mu']
        self.prior_sigma = priors['prior_sigma']
        self.prior_log_sigma = math.log(self.prior_sigma)


        self.W_mu = nn.Parameter(torch.empty((1)))
        self.W_rho = nn.Parameter(torch.empty((1)))

        if self.use_bias:
            self.bias_mu = nn.Parameter(torch.empty((1)))
            self.bias_rho = nn.Parameter(torch.empty((1)))
        else:
            self.register_parameter('bias_mu', None)
            self.register_parameter('bias_rho', None)

        self._reset_parameters()

    def _reset_parameters(self):

        self.W_mu.data.normal_(self.prior_mu, self.prior_sigma)
        self.W_rho.data.fill_(self.prior_log_sigma)

        if self.use_bias:
            self.bias_mu.data.uniform_(self.prior_mu, self.prior_sigma)
            self.bias_rho.data.fill_(self.prior_log_sigma)

    def forward(self, input, sample=True):
        if self.training or sample:
            W_eps = torch.empty((self.out_features, self.in_features)).normal_(0, 1).cuda()
            self.W_sigma = torch.log1p(torch.exp(self.W_rho))
            weight = self.W_mu + W_eps * self.W_sigma

            if self.use_bias:
                bias_eps = torch.empty((self.out_features)).normal_(0, 1).cuda()
                self.bias_sigma = torch.log1p(torch.exp(self.bias_rho))
                bias = self.bias_mu + bias_eps * self.bias_sigma
            else:
                bias = None
        else:
            weight = self.W_mu
            bias = self.bias_mu if self.use_bias else None

        return F.linear(input, weight, bias)

    #KL loss
    def _calculate_kl(self, mu_q, sig_q, mu_p, sig_p):
        kl = 0.5 * (2 * torch.log(sig_p / sig_q) - 1 + (sig_q / sig_p).pow(2) + ((mu_p - mu_q) / sig_p).pow(2)).sum()
        return kl

    def kl_loss(self):
        kl = self._calculate_kl(self.prior_mu, self.prior_sigma, self.W_mu, self.W_sigma)
        if self.use_bias:
            kl += self._calculate_kl(self.prior_mu, self.prior_sigma, self.bias_mu, self.bias_sigma)
        return kl


if __name__ == "__main__":
    #for debug  
    x =  torch.rand(10, 1, 512).cuda()
    k_size = 3 
    bconv = BayesConv1d(in_channels=1, out_channels=1, kernel_size=k_size).cuda()
    out = bconv(x)
    print(bconv.kl_loss())
    print(out.shape)

    x =  torch.rand(10, 512).cuda()
    bfc = BayesFC(512, 512).cuda()
    out = bfc(x)
    print(bfc.kl_loss())
    print(out.shape)
