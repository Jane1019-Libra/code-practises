import torch
import torch.nn as nn
import math

class layerNormalization(nn.Module):
    def __init__(self, normalized_shape, eps = 1e-5, elementwise_affine=True):
        super().__init__() 
        self.normalized_shape = normalized_shape
        self.eps = eps
        self.gamma = None
        self.beta = None
        if elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(normalized_shape))
            self.beta = nn.Parameter(torch.zeros(normalized_shape))

    def forward(self, x):
        dims = torch.arange(-len(self.normalized_shape), 0)
        x = (x - torch.mean(x, dim = dims, keepdim = True)) * torch.rsqrt(torch.var(x, dim = -1, keepdim = True, unbiased=False) + self.eps)
        if self.gamma is not None:
            x = self.gamma * x + self.beta
        return x