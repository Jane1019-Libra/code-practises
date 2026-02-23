import torch
import torch.nn as nn
import math

class RoteryPositionEncoding(nn.Module):
    f'''
    theta = pos / 10000^(2 * i / d)
    '''
    def __init__(self, d_model, max_len=5000, base=10000.0) -> None:
        super().__init__()
        freqs = 1 / (
            base ** torch.arange(0, d_model, 2, dtype=torch.float) / d_model
        ) # (d_model / 2, )
        pos = torch.arange(0, max_len, dtype=torch.float)   # (max_len, )
        angles = pos.unsqueeze(1) * freqs   # (max_len, d_model / 2)
        
        self.register_buffer("sin_cached", angles.sin().unsqueeze(0))
        self.register_buffer("cos_cached", angles.cos().unsqueeze(0))


    def forward(self, q: torch.Tensor, k: torch.Tensor):
        seq_len = q.shape[1]
        sin = self.sin_cached[:, :seq_len, :]
        cos = self.cos_cached[:, :seq_len, :]

        return self._apply_rope(q, sin, cos), self._apply_rope(k, sin, cos)
    

    def _apply_rope(self, x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor):
        even_index = x[..., ::2]
        odd_index = x[..., 1::2]

        new_even = even_index * cos + odd_index * sin
        new_odd = even_index * sin + odd_index * cos
        # torch.stack creates a new dim
        # torch.flatten flattens from start dim 
        # Reduce from the innermost dim
        return torch.stack([new_even, new_odd], dim=-1).flatten(-2)