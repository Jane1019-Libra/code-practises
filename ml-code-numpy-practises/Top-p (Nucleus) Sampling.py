import numpy as np

def top_p_sampling(logits: list[float], p: float) -> list[float]:
    """
    Apply top-p (nucleus) sampling to filter a probability distribution.
    
    Args:
        logits: Raw unnormalized scores for each token
        p: Cumulative probability threshold (0 < p <= 1)
    
    Returns:
        Filtered and renormalized probability distribution as a list of floats
    """
    exp_new_logits = np.exp(np.array(logits))
    prob = exp_new_logits / np.sum(exp_new_logits)
    idx = np.argsort(-prob, kind="stable")
    new_prob = prob[idx]
    cum = np.cumsum(new_prob)
    latestidx = np.argmax(cum >= p)
    ans = np.zeros_like(exp_new_logits)
    ans[idx[:latestidx + 1]] = prob[idx[:latestidx + 1]] / cum[latestidx]
    return np.round(ans, 4).tolist()