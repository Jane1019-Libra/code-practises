import torch.nn as nn

class TransformerDecoderBlock(nn.Module):
    def __init__(self, d_model, num_heads, ffn_hidden_dim, dropout = 0.1):
        super().__init__()
        self.self_attention = MultiHeadAttention(d_model, num_heads)
        self.dropout1 = nn.Dropout(dropout)
        self.ln1 = nn.LayerNorm(d_model)
        self.cross_attention = MultiHeadAttention(d_model,num_heads)
        self.dropout2 = nn.Dropout(dropout)
        self.ln2 = nn.LayerNorm(d_model)

        self.ffn = nn.Sequential(
            nn.Linear(d_model, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, d_model)   
        )
        self.dropout3 = nn.Dropout(dropout)
        self.ln3 = nn.LayerNorm(d_model)
    
    def forward(self, x, encoder_output, self_attention_mask = None, cross_attention_mask = None):
        self_attention_output = self.self_attention(x, self_attention_mask)
        x = self.ln1(x + self.dropout1(self_attention_output))

        cross_attn_output = self.cross_attention(x)
        x = self.ln2(x + self.dropout2(cross_attn_output))

        ffn_output = self.ffn(x)
        x = self.ln3(x + self.dropout3(ffn_output))
        return x


