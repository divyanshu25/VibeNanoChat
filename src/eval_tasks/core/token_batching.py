"""
Token batching utilities for CORE evaluation tasks.

Handles tokenization, sequence batching, and identifying evaluation regions
for different task types.
"""

from typing import List, Tuple

import torch


def find_common_length(
    token_sequences: List[List[int]], direction: str = "left"
) -> int:
    """
    Find the length of the common prefix or suffix across token sequences.

    Used to identify:
    - Common prefix in multiple choice (same context, different continuations)
    - Common suffix in schema tasks (different contexts, same continuation)

    Args:
        token_sequences: List of token ID sequences to compare
        direction: 'left' for prefix, 'right' for suffix

    Returns:
        Number of tokens in the common prefix/suffix
    """
    min_len = min(len(seq) for seq in token_sequences)
    indices = {"left": range(min_len), "right": range(-1, -min_len - 1, -1)}[direction]

    # Find the first position where the token sequences differ
    for i, idx in enumerate(indices):
        token = token_sequences[0][idx]
        if not all(seq[idx] == token for seq in token_sequences):
            return i
    return min_len


def stack_sequences(tokens: List[List[int]], pad_token_id: int) -> torch.Tensor:
    """
    Stack a list of token sequences into a batch tensor with right-padding.

    Args:
        tokens: List of token ID sequences of varying lengths
        pad_token_id: Token ID to use for padding shorter sequences

    Returns:
        Tensor of shape (batch_size, max_sequence_length) with padded sequences
    """
    bsz, seq_len = len(tokens), max(len(x) for x in tokens)
    input_ids = torch.full((bsz, seq_len), pad_token_id, dtype=torch.long)
    for i, x in enumerate(tokens):
        input_ids[i, : len(x)] = torch.tensor(x, dtype=torch.long)
    return input_ids


def batch_sequences_mc(
    tokenizer, prompts: List[str]
) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    Tokenize and prepare multiple choice prompts for batched evaluation.

    In multiple choice tasks, all prompts share the same context (common prefix)
    but have different answer continuations. We identify where continuations start.

    Args:
        tokenizer: Tokenizer instance with __call__ and get_bos_token_id methods
        prompts: List of complete prompts, one per answer choice

    Returns:
        tokens: List of tokenized sequences
        start_indices: Start index of continuation for each prompt (all same value)
        end_indices: End index (length) of each prompt
    """
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # Find where the answer continuations start (end of common prefix)
    answer_start_idx = find_common_length(tokens, direction="left")
    start_indices = [answer_start_idx] * len(prompts)
    end_indices = [len(x) for x in tokens]
    return tokens, start_indices, end_indices


def batch_sequences_schema(
    tokenizer, prompts: List[str]
) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    Tokenize and prepare schema prompts for batched evaluation.

    In schema tasks, prompts have different contexts but the same continuation
    (common suffix). We identify where the continuation starts in each prompt.

    Args:
        tokenizer: Tokenizer instance with __call__ and get_bos_token_id methods
        prompts: List of complete prompts, one per context option

    Returns:
        tokens: List of tokenized sequences
        start_indices: Start index of continuation for each prompt
        end_indices: End index (length) of each prompt
    """
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    # Find the length of the common continuation suffix
    suffix_length = find_common_length(tokens, direction="right")
    end_indices = [len(x) for x in tokens]
    start_indices = [ei - suffix_length for ei in end_indices]
    return tokens, start_indices, end_indices


def batch_sequences_lm(
    tokenizer, prompts: List[str]
) -> Tuple[List[List[int]], List[int], List[int]]:
    """
    Tokenize and prepare language modeling prompts for evaluation.

    In LM tasks, we receive two prompts: without and with the continuation.
    This allows us to identify exactly which tokens need to be predicted.

    Args:
        tokenizer: Tokenizer instance with __call__ and get_bos_token_id methods
        prompts: List of exactly 2 prompts [without_continuation, with_continuation]

    Returns:
        tokens: List with single tokenized sequence (with continuation)
        start_indices: List with single start index of continuation
        end_indices: List with single end index (length of full prompt)
    """
    tokens = tokenizer(prompts, prepend=tokenizer.get_bos_token_id())
    tokens_without, tokens_with = tokens
    start_idx, end_idx = len(tokens_without), len(tokens_with)

    # Validate that prompt_without is a proper prefix of prompt_with
    assert (
        start_idx < end_idx
    ), "prompt without is supposed to be a prefix of prompt with"
    assert (
        tokens_without == tokens_with[:start_idx]
    ), "prompt without is supposed to be a prefix of prompt with"

    # Return batch size of 1 (only the with_continuation prompt)
    return [tokens_with], [start_idx], [end_idx]
