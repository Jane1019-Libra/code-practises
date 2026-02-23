import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class MultiQueryAttention(nn.Modules):
    def __init__(self, d_model, num_heads, kv_groups):
        super().__init__()

        assert d_model % num_heads == 0
        self.kv_groups = kv_groups
        self.d_k = d_model // num_heads
        self.d_model = d_model
        self.num_heads = num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, self.d_k * self.kv_groups)
        self.W_V = nn.Linear(d_model, self.d_k * self.kv_groups)

        self.W_O = nn.Linear(d_model, d_model)
    
    def forward(self, x, mask = None):
        batch_size, seq_len, _ = x.size()

        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(batch_size, seq_len, self.num_heads, self.d_k).transpose(1,2)

        K = K.view(batch_size, seq_len, self.kv_groups, self.d_k).transpose(1,2)
        V = V.view(batch_size, seq_len, self.kv_groups, self.d_k).transpose(1,2)

        if self.kv_groups != self.num_heads:
            repeat = self.num_heads // self.kv_groups
            K = K.repeat_interleave(repeat, dim = 1)
            V = V.repeat_interleave(repeat, dim = 1)
            
        score = torch.matmul(Q, K.transpose(-1,-2)) / torch.sqrt(self.d_k)

        if mask is not None:
            if len(mask.size()) == 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            score = score.masked_fill(mask == 0, float('-int'))
        
        attention_score = torch.matmul(F.softmax(score, dim = -1), V)

        attention_score = attention_score.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)
        return self.W_O(torch.matmul(attention_score, V))


