import os
import torch
import torch.nn as nn

class FeatureExtractor(nn.Module):
    def __init__(self, submodule, extracted_layers):
        super(FeatureExtractor, self).__init__()
        self.submodule = submodule
        self.extracted_layers = extracted_layers
 
    def forward(self, x):
        outputs = []
        #print('---------',self.submodule._modules.items())
        for name, module in self.submodule._modules.items():
            #if "fc" in name:
            #    x = x.view(x.size(0), -1)
            #print(module)
            #x = module(x)
            #print('name', name)
            if name in self.extracted_layers:
                print('name', name)
                x = module(x)
                outputs.append(x)
                
        return outputs
        