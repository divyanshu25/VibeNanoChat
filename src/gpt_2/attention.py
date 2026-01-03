import torch
import torch.nn as nn
import math
import torch.nn.functional as F


class CausalSelfAttention(nn.Module):
    def __init__(self, config):
        super().__init__()
        assert config.n_embed % config.n_head == 0

        # Setup for GQA vs MHA
        self.n_head = config.n_head
        self.n_embed = config.n_embed
        self.head_dim = config.n_embed // config.n_head

        # If n_kv_head is None or equal to n_head, use MHA; otherwise use GQA
        self.n_kv_head = (
            config.n_kv_head if config.n_kv_head is not None else config.n_head
        )
        assert (
            config.n_head % self.n_kv_head == 0
        ), "n_head must be divisible by n_kv_head"

        # For GQA: Q has n_head, but K and V have n_kv_head
        kv_dim = self.head_dim * self.n_kv_head

        # key, query, value projections
        # Q: n_embed, K: kv_dim, V: kv_dim
        self.c_attn = nn.Linear(config.n_embed, config.n_embed + 2 * kv_dim)

        # output projection
        self.c_proj = nn.Linear(config.n_embed, config.n_embed)
        self.c_proj.NANOGPT_SCALE_INIT = 1

        self.register_buffer(
            "bias",
            torch.tril(torch.ones(config.block_size, config.block_size)).view(
                1, 1, config.block_size, config.block_size
            ),
        )

    def forward(self, x):
        B, T, C = x.size()  # batch size, sequence length, embedding dimension
        # calculate query, key, values for all heads in batch and move head forward to be the batch dim
        # nh is number of heads, hs is head size and C is embedding dimension = nh * hs
        # e.g in GPT-2 (124M) n_head = 12 and hs = 64, so nh*hs = 768 channels in the Transfromer

        qkv = self.c_attn(x)

        # Split into Q, K, V with potentially different sizes for GQA
        kv_dim = self.head_dim * self.n_kv_head
        q, k, v = qkv.split([self.n_embed, kv_dim, kv_dim], dim=2)

        # Reshape Q with n_head
        q = q.view(B, T, self.n_head, self.head_dim).transpose(
            1, 2
        )  # (B, n_head, T, head_dim)

        # Reshape K and V with n_kv_head
        k = k.view(B, T, self.n_kv_head, self.head_dim).transpose(
            1, 2
        )  # (B, n_kv_head, T, head_dim)
        v = v.view(B, T, self.n_kv_head, self.head_dim).transpose(
            1, 2
        )  # (B, n_kv_head, T, head_dim)

        # For GQA: repeat K and V heads to match Q heads
        if self.n_kv_head < self.n_head:
            n_rep = self.n_head // self.n_kv_head
            # Repeat each KV head n_rep times
            k = k.repeat_interleave(n_rep, dim=1)  # (B, n_head, T, head_dim)
            v = v.repeat_interleave(n_rep, dim=1)  # (B, n_head, T, head_dim)

        # att = (q @ k.transpose(-2, -1)) * (1 / math.sqrt(k.size(-1)))
        # att = att.masked_fill(self.bias[:, :, :T, :T] == 0, float("-inf"))
        # att = F.softmax(att, dim=-1)
        # y = att @ v  # (B, nh, T, T) x (B, nh, T, hs) -> (B, nh, T, hs)

        # flash attention
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True)

        y = (
            y.transpose(1, 2).contiguous().view(B, T, C)
        )  # re-assemble all head outputs side by side
        # output projection
        y = self.c_proj(y)
        return y
