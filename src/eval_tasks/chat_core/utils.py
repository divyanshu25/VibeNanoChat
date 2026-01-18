"""
Utility functions for ChatCORE evaluation tasks.

Provides helper functions to:
- Render prompts from conversations
- Setup evaluators with common configurations
- Register tasks with the evaluator

Performance Note:
==================
The ChatCORE evaluator uses KV (Key-Value) caching for efficient generation:
- Without KV cache: Each token requires reprocessing all previous tokens → O(N²)
- With KV cache: Each token reuses cached attention keys/values → O(N)
- Result: 5-10x faster generation with ~2x memory usage

This is implemented transparently in the evaluator's generate methods.
"""

from typing import Dict, List, Optional


def render_conversation_for_completion(conversation: Dict, tokenizer) -> List[int]:
    """
    Render a conversation into tokens for completion using special token format.

    Takes a conversation dict with messages and renders it into a prompt
    that ends right before where the assistant should respond, using the
    special token format that matches training:
    <|bos|><|user_start|>...<|user_end|><|assistant_start|>

    Args:
        conversation: Dict with 'messages' key containing list of messages.
                     Each message has 'role' and 'content' keys.
        tokenizer: Tokenizer with encode method (supports allowed_special="all")

    Returns:
        List of token IDs representing the prompt

    Example:
        >>> conversation = {
        ...     'messages': [
        ...         {'role': 'user', 'content': 'What is 2+2?'},
        ...         {'role': 'assistant', 'content': ...}  # Will be predicted
        ...     ]
        ... }
        >>> tokens = render_conversation_for_completion(conversation, tokenizer)
        # Returns tokens for: "<|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>"
    """
    messages = conversation["messages"]
    formatted_parts = ["<|bos|>"]  # Start with beginning-of-sequence token

    for i, msg in enumerate(messages):
        role = msg["role"].lower()
        content = msg["content"]

        # Extract text content from structured or string format
        if isinstance(content, str):
            text_content = content
        else:
            # If content is structured (list of parts), extract text
            text_parts = [p["text"] for p in content if p.get("type") == "text"]
            text_content = "".join(text_parts)

        # For assistant messages, add the start token but not the content
        # (we only want to render up to where assistant should respond)
        if role == "assistant":
            formatted_parts.append("<|assistant_start|>")
            break
        elif role == "user":
            # Add user message with special tokens
            formatted_parts.append(f"<|user_start|>{text_content}<|user_end|>")
        else:
            # Handle other roles (system, etc.) as user for consistency with training
            formatted_parts.append(f"<|user_start|>{text_content}<|user_end|>")

    # Combine into final prompt text
    prompt_text = "".join(formatted_parts)

    # Encode to tokens with special token support
    # The allowed_special="all" parameter tells the tokenizer to recognize
    # our custom special tokens (<|bos|>, <|user_start|>, etc.)
    tokens = tokenizer.encode(prompt_text, allowed_special="all")
    return tokens


def setup_gsm8k_task(
    evaluator,
    tokenizer,
    split: str = "test",
    cache_dir: Optional[str] = "/sensei-fs/users/divgoyal/nanochat_midtraining_data",
):
    """
    Setup GSM8K task with the evaluator.

    Args:
        evaluator: ChatCoreEvaluator instance
        tokenizer: Tokenizer to use for encoding
        split: Dataset split ('train' or 'test')
        cache_dir: Directory to cache the downloaded HuggingFace dataset
    """
    from .gsm8k import evaluate_gsm8k, load_gsm8k_from_hf

    def load_fn(max_examples=None):
        """Load GSM8K data."""
        return load_gsm8k_from_hf(
            split=split, max_examples=max_examples, cache_dir=cache_dir
        )

    def eval_fn(example, generated_text):
        """Evaluate a GSM8K prediction."""
        ground_truth_answer = example["answer"]
        return evaluate_gsm8k(ground_truth_answer, generated_text)

    def render_fn(example):
        """Render GSM8K conversation to prompt tokens."""
        conversation = example["conversation"]
        return render_conversation_for_completion(conversation, tokenizer)

    # Register with evaluator
    evaluator.register_task(
        "GSM8K",
        {
            "load_fn": load_fn,
            "eval_fn": eval_fn,
            "render_fn": render_fn,
        },
    )
