#Values grow larger each layer → gradients explode, parameter updates go out of control
#Values shrink each layer → gradients vanish, the network stops learning
#It forcibly pulls each layer's output back to a stable distribution
#Then learnable parameters γ and β give the network back some flexibility to shift and scale as needed.

import torch
from torch import nn


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps = 1e-5, elementwise_affine=True):
        super().__init__()
        self.eps = eps
        self.normalized_shape = normalized_shape
        self.elementwise_affine = elementwise_affine

        if self.elementwise_affine:
            self.gamma = nn.Parameter(torch.ones(self.normalized_shape))
            self.beta = nn.Parameter(torch.zeros(self.normalized_shape))

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        assert x.shape[-len(self.normalized_shape):] == self.normalized_shape
        dims = tuple(range(-len(self.normalized_shape), 0))

        var, mean = torch.var_mean(x, dim=dims, keepdim=True, unbiased=False)
        x_norm = (x - mean) * torch.rsqrt(var + self.eps)

        if self.elementwise_affine:
            return self.gamma * x_norm + self.beta
        return x_norm