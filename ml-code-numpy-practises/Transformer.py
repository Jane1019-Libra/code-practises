import torch

d_model = 512
num_heads = 8
ffn_hidden_dim = 2048
dropout = 0.1
batch_size = 2
seq_len = 10

encoder_block = TransformerEncoderBlock(d_model, num_heads, ffn_hidden_dim, dropout)
decoder_block = TransformerDecoderBlock(d_model, num_heads, ffn_hidden_dim, dropout)

x_encoder = torch.randn(batch_size, seq_len, d_model) 
x_decoder = torch.randn(batch_size, seq_len, d_model)

encoder_output = encoder_block(x_encoder)
decoder_output = decoder_block(
    x=x_decoder,
    encoder_output=encoder_output,
    self_attention_mask=None,
    cross_attention_mask=None
)

print(decoder_output.shape) 