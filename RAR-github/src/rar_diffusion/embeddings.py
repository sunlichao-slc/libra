from __future__ import annotations

import numpy as np
import torch


def ensure_3d_embedding(embedding: torch.Tensor | np.ndarray) -> torch.Tensor:
    """Shape encoder hidden states to [batch, seq_len, dim] for UNet cross-attn."""
    if isinstance(embedding, np.ndarray):
        embedding = torch.tensor(embedding, dtype=torch.float32)
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0).unsqueeze(1)
    elif embedding.ndim == 2:
        embedding = embedding.unsqueeze(0)
    elif embedding.ndim > 3:
        while embedding.ndim > 3 and embedding.size(0) == 1:
            embedding = embedding.squeeze(0)
        if embedding.ndim > 3:
            b = embedding.size(0)
            embedding = embedding.view(b, -1, embedding.size(-1))
    return embedding
