import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiHeadAttention(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0

        self.d_model = d_model
        self.num_heads = num_heads
        self.d_k = d_model // num_heads
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)
        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask=None):
        batch_size, seq_len, _ = x.size()

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        K = K.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)
        scores = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.d_k)

        if mask is not None:
            if mask.dim() == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            scores = scores.mask_fill(mask == 0, float('-inf'))

        attn_scores = F.softmax(scores, dim = -1)

        output = torch.matmul(attn_scores, V)
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_O(output)