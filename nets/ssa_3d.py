# encoding: utf-8
"""
3D Uncertainty Spatial Attention with Spectral Convolution
Author: Jason.Fang
Update time: 25/06/2021
"""

import torch
from torch.autograd import Variable
import torch.nn.functional as F
from torch import nn
from torch import Tensor
from torch.nn import Parameter

def l2normalize(v, eps=1e-12):
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    def __init__(self, module, name='weight', power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        self._make_params()

    def _update_u_v(self):
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

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


#Spatial-wise Spectral Attention
class SpatialSpectralAttention(nn.Module): 
    def __init__(self, in_ch, k, k_size=3):
        super(SpatialSpectralAttention, self).__init__()

        self.in_ch = in_ch
        self.out_ch = in_ch
        self.mid_ch = in_ch // k

        #print('Num channels:  in    out    mid')
        #print('               {:>4d}  {:>4d}  {:>4d}'.format(self.in_ch, self.out_ch, self.mid_ch))

        self.f = nn.Sequential(
            nn.Conv3d(self.in_ch, self.mid_ch, (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(self.mid_ch),
            nn.ReLU())
        self.g = nn.Sequential(
            nn.Conv3d(self.in_ch, self.mid_ch, (1, 1, 1), (1, 1, 1)),
            nn.BatchNorm3d(self.mid_ch),
            nn.ReLU())
        self.h = nn.Conv3d(self.in_ch, self.mid_ch, (1, 1, 1), (1, 1, 1))
        self.v = nn.Conv3d(self.mid_ch, self.out_ch, (1, 1, 1), (1, 1, 1))

        #self.softmax = nn.Softmax(dim=-1)
        self.spe_norm = SpectralNorm(nn.Conv2d(1, 1, k_size, stride=1, padding=(k_size - 1) // 2 ))

        for conv in [self.f, self.g, self.h]: 
            conv.apply(weights_init)
        self.v.apply(constant_init)

    def forward(self, x):
        B, C, D, H, W = x.shape

        f_x = self.f(x).view(B, self.mid_ch, D * H * W)  # B * mid_ch * N, where N = D*H*W
        g_x = self.g(x).view(B, self.mid_ch, D * H * W)  # B * mid_ch * N, where N = D*H*W
        h_x = self.h(x).view(B, self.mid_ch, D * H * W)  # B * mid_ch * N, where N = D*H*W

        z = torch.bmm(f_x.permute(0, 2, 1), g_x)  # B * N * N, where N = D*H*W
        #attn = self.softmax((self.mid_ch ** -.50) * z)
        attn = self.spe_norm(z.unsqueeze(1)).squeeze()

        z = torch.bmm(attn, h_x.permute(0, 2, 1))  # B * N * mid_ch, where N = D*H*W
        z = z.permute(0, 2, 1).view(B, self.mid_ch, D, H, W)  # B * mid_ch * D * H * W

        z = self.v(z)
        x = torch.add(z, x) # z + x
        return x

## Kaiming weight initialisation
def weights_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.kaiming_normal_(module.weight.data)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass
def constant_init(module):
    if isinstance(module, nn.ReLU):
        pass
    if isinstance(module, nn.Conv2d) or isinstance(module, nn.ConvTranspose2d):
        nn.init.constant_(module.weight.data, 0.0)
        nn.init.constant_(module.bias.data, 0.0)
    elif isinstance(module, nn.BatchNorm2d):
        pass

if __name__ == "__main__":
    #for debug  
    x =  torch.rand(2, 512, 10, 10, 10).cuda()
    ssa = SpatialSpectralAttention(in_ch=512, k=2, k_size=5).cuda()
    out = ssa(x)
    print(out.shape)