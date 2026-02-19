import numpy as np

def compute_qkv(X, W_q, W_k, W_v):
    """Compute Query, Key, Value matrices from input X and weight matrices."""
    Q = np.dot(X, W_q)
    K = np.dot(X, W_k)
    V = np.dot(X, W_v)
    return Q, K, V

def self_attention(Q, K, V):
    """
    Compute scaled dot-product self-attention.
    
    Args:
        Q: Query matrix of shape (seq_len, d_k)
        K: Key matrix of shape (seq_len, d_k)
        V: Value matrix of shape (seq_len, d_v)
    
    Returns:
        Attention output of shape (seq_len, d_v)
    """
    # Your code here
    scores = (Q @ K.T) / np.sqrt(np.shape(K)[1])
    row_max = np.max(scores, axis = -1, keepdims = True)
    scores_safety = scores - row_max
    scores_safety_exp = np.exp(scores_safety)
    soft_max = scores_safety_exp / np.sum(scores_safety_exp, axis = -1, keepdims = True)
    return soft_max @ V
