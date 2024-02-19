import numpy as np

import numpy as np
import torch


def create_masks(inp, device=None):
    """
    Create both padding mask and attention mask for the input sequence.

    Args:
    inp: Input sequence tensor (PyTorch).

    Returns:
    mask: Combined mask tensor (PyTorch).
    """
    seq_np = inp.cpu().numpy() if inp.is_cuda else inp.numpy()

    def get_padding_mask(seq):
        padding_mask = (seq == 0).astype(float)
        # Add extra dimensions to add the padding to the attention logits.
        return padding_mask[:, np.newaxis, np.newaxis, :]  # (batch_size, 1, 1, seq_len)

    def attention_mask(size):
        mask = 1 - np.tril(np.ones((size, size)))
        return mask  # (seq_len, seq_len)

    att_mask = attention_mask(seq_np.shape[1])
    padding_mask = get_padding_mask(seq_np)
    mask_np = np.maximum(padding_mask, att_mask[np.newaxis, :, :])

    mask = torch.tensor(mask_np, dtype=torch.float32)

    return mask.to(device)