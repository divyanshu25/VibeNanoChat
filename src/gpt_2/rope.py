"""
RoPE - Rotary Position Embeddings
Relative position encoding via rotation in complex space.
Used in nanochat and modern LLMs (LLaMA, GPT-NeoX, PaLM, etc.)
"""

import torch


def apply_rotary_emb(x, cos, sin):
    """
    Apply rotary position embeddings to input tensor.

    RoPE encodes relative positions by rotating query and key vectors.
    Instead of adding position info, it rotates pairs of dimensions.

    Args:
        x: Input tensor of shape (B, T, n_head, head_dim)
        cos: Cosine component of shape (1, T, 1, head_dim//2)
        sin: Sine component of shape (1, T, 1, head_dim//2)

    Returns:
        Rotated tensor of same shape as input
    """
    assert x.ndim == 4, "Expected 4D tensor (B, T, n_head, head_dim)"

    # Split head_dim into two halves
    d = x.shape[3] // 2
    x1, x2 = x[..., :d], x[..., d:]  # This will split the head_dim into two halves

    # Apply rotation to pairs of dimensions
    # This is equivalent to complex number rotation: e^(-iθ) * z
    # Note: This uses negative rotation (matching nanochat implementation)
    # 2D rotation matrix: [cos sin; -sin cos]
    y1 = x1 * cos + x2 * sin
    y2 = x1 * (-sin) + x2 * cos

    return torch.cat([y1, y2], dim=3)  # shape: (B, T, n_head, head_dim)


def precompute_rotary_embeddings(
    seq_len, head_dim, base=10000, device=None, dtype=torch.bfloat16
):
    """
    Precompute cos and sin tensors for RoPE.

    Creates rotation matrices for all positions in sequence.
    Each position gets a different rotation angle based on its position index.

    Args:
        seq_len: Maximum sequence length to precompute for
        head_dim: Dimension of attention head (must be even)
        base: Base for exponential decay (10000 in original, can be higher for longer contexts)
        device: Device to create tensors on (auto-detected if None)
        dtype: Data type for tensors (bfloat16 recommended for efficiency)

    Returns:
        tuple: (cos, sin) tensors of shape (1, seq_len, 1, head_dim//2)
               Ready for broadcasting in attention computation
    """
    assert head_dim % 2 == 0, "head_dim must be even for RoPE"

    # Compute inverse frequencies for exponential decay
    # Lower frequencies for higher dimensions -> longer-range position info
    # Shape: (head_dim // 2,)
    inv_freq = 1.0 / (
        base
        ** (torch.arange(0, head_dim, 2, dtype=torch.float32, device=device) / head_dim)
    )

    # Create position indices [0, 1, 2, ..., seq_len-1]
    # Shape: (seq_len,)
    t = torch.arange(seq_len, dtype=torch.float32, device=device)  # shape: (seq_len,)

    # Outer product: position × frequency = rotation angle for each (time, channel) pair
    # Shape: (seq_len, head_dim // 2)
    freqs = torch.outer(t, inv_freq)  # shape: (seq_len, head_dim // 2)

    # Convert to cos/sin for efficient rotation
    cos, sin = freqs.cos(), freqs.sin()  # shape: (seq_len, head_dim // 2)

    # Convert to target dtype (bfloat16 for memory efficiency)
    cos, sin = cos.to(dtype), sin.to(dtype)  # shape: (seq_len, head_dim // 2)

    # Add batch and head dimensions for broadcasting: (1, seq_len, 1, head_dim//2)
    # This allows efficient broadcasting over (B, T, n_head, head_dim) tensors
    cos = cos[None, :, None, :]  # (1, seq_len, 1, head_dim//2)
    sin = sin[None, :, None, :]  # (1, seq_len, 1, head_dim//2)

    return cos, sin
