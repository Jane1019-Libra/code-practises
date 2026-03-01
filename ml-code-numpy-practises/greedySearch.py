import torch
import torch.nn.functional as F


def greedy_search(model_logits, max_len = 20, eos_token_id = 2):
    batch_size = model_logits.size(0)

    generated_sequence = torch.zeros(batch_size, max_len, dtype = torch.long)

    for t in range(max_len):
        current_logits = model_logits[:, t, :]
        probs = F.softmax(current_logits, dim = -1)
        next_token = torch.argmax(probs, dim = -1)
        generated_sequence[:, t] = next_token

        if (next_token == eos_token_id).all():
            break
    return generated_sequence


