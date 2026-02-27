import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class KVCache(nn.Module):
    def __init__(self, d_model, num_heads):
        super().__init__()
        self.num_heads = num_heads
        self.d_model = d_model
        self.d_k = d_model // num_heads
        
        self.W_Q = nn.Linear(d_model, d_model)
        self.W_K = nn.Linear(d_model, d_model)
        self.W_V = nn.Linear(d_model, d_model)

        self.W_O = nn.Linear(d_model, d_model)
    

    def forward(self, x, mask = None, past_key_value = None):
        batch_size, seq_len, _ = x.size()
        Q = self.W_Q(x)
        K = self.W_K(x)
        V = self.W_V(x)

        Q = Q.view(batch_size,seq_len, self.num_heads, self.d_k).transpose(1,2)
        K = K.view(batch_size,seq_len, self.num_heads, self.d_k).transpose(1,2)
        V = V.view(batch_size,seq_len, self.num_heads, self.d_k).transpose(1,2)

        if past_key_value is not None:
            key, value = past_key_value
            K = torch.cat([key, K], dim = 2)
            V = torch.cat([value, V], dim = 2)
        present_key_value = (K,V)

        scores = torch.matmul(Q, K.transpose(-1,-2)) / math.sqrt(self.d_k)
        if mask is not None:
            if len(mask.size()) != 2:
                mask = mask.unsqueeze(0).unsqueeze(1)
            scores = scores.masked_fill(mask == 0, float('-inf'))
        
        softmax = F.softmax(scores, dim = -1)
        output = torch.matmul(softmax, V)
        output = output.transpose(1,2).contiguous().view(batch_size, seq_len, self.d_model)

        return self.W_O(output), present_key_value
