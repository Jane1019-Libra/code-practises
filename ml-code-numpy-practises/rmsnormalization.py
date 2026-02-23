import torch
import torch.nn as nn

class RMSNorm(nn.module):
    def __init(self, d_model: int, eps: float = 1e-5, elementwise_affline: bool = True);
        super().__init__()
        self.eps = eps
        self.elementwise_affline = elementwise_affline

        if elementwise_affline:
            self.weight = nn.Parameter(torch.ones(d_model))
        else:
            self.weight = None

    def forward(self, x:torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(x.pow(2).mean(-1, keepdim = True)) + self.eps
        x_hat = x / rms
        if self.weight is None:
            return x_hat
        else:
            return self.weight * x_hat