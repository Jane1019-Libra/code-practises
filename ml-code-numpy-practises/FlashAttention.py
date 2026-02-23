import torch
import torch.nn as nn
import torch.nn.functional as F
import math


class FlashAttention(nn.Module):
    def __init__(self, embed_dim: int, num_heads: int):
        super(FlashAttention, self).__init__()
        assert embed_dim % num_heads == 0
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = 1.0 / math.sqrt(self.head_dim)


        self.q_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.k_proj = nn.Linear(embed_dim, embed_dim, bias=False)
        self.v_proj = nn.Linear(embed_dim, embed_dim, bias=False)

        self.o_proj = nn.Linear(embed_dim, embed_dim)

    def forward(self, x: torch.Tensor, attn_mask: torch.Tensor = None):
        batch_size, seq_len, _ = x.size()

        q = self.q_proj(x)
        k = self.k_proj(x)
        v = self.v_proj(x)

        Q = q.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        K = k.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)
        V = v.view(batch_size, seq_len, self.num_heads, self.head_dim).transpose(1,2)

        