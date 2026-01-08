"""
Simple tokenizer wrapper for CORE evaluation.

Provides a consistent interface for tokenization across evaluation tasks,
wrapping tiktoken-based encoders with convenience methods.
"""

from typing import List, Optional

import torch


class SimpleTokenizer:
    """
    Tokenizer wrapper to provide consistent interface for CORE evaluation.

    Wraps a tiktoken encoding object and adds convenience methods needed
    by the evaluation functions (BOS token handling, batch encoding, etc.).
    """

    def __init__(self, enc, bos_token_id: int = 50256):
        """
        Initialize tokenizer wrapper.

        Args:
            enc: tiktoken Encoding object (e.g., tiktoken.get_encoding("gpt2"))
            bos_token_id: Beginning-of-sequence token ID (default: 50256 for GPT-2)
        """
        self.enc = enc
        self.bos_token_id = bos_token_id

    def __call__(
        self, texts: str | List[str], prepend: Optional[int] = None
    ) -> List[int] | List[List[int]]:
        """
        Encode text(s) to token IDs.

        Args:
            texts: Single string or list of strings to encode
            prepend: Optional token ID to prepend to each sequence (typically BOS token)

        Returns:
            If single string input: List of token IDs
            If list input: List of lists of token IDs
        """
        if isinstance(texts, str):
            texts = [texts]

        results = []
        for text in texts:
            tokens = self.enc.encode(text, allowed_special="all")
            if prepend is not None:
                tokens = [prepend] + tokens
            results.append(tokens)

        # If single string was passed, return flat list (not nested)
        if len(results) == 1:
            return results[0]
        return results

    def get_bos_token_id(self) -> int:
        """Return the beginning-of-sequence token ID."""
        return self.bos_token_id

    def decode(self, tokens: List[int] | torch.Tensor) -> str:
        """
        Decode token IDs back to text string.

        Args:
            tokens: List of token IDs or tensor of token IDs

        Returns:
            Decoded text string
        """
        if isinstance(tokens, torch.Tensor):
            tokens = tokens.tolist()
        return self.enc.decode(tokens)
