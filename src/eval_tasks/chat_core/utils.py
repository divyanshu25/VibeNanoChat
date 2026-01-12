"""
Utility functions for ChatCORE evaluation tasks.

Provides helper functions to:
- Render prompts from conversations
- Setup evaluators with common configurations
- Register tasks with the evaluator
"""

from typing import Dict, List


def render_conversation_for_completion(conversation: Dict, tokenizer) -> List[int]:
    """
    Render a conversation into tokens for completion.

    Takes a conversation dict with messages and renders it into a prompt
    that ends right before where the assistant should respond.

    Args:
        conversation: Dict with 'messages' key containing list of messages.
                     Each message has 'role' and 'content' keys.
        tokenizer: Tokenizer with encode method

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
        # Returns tokens for: "User: What is 2+2?\nAssistant:"
    """
    # Simple chat template rendering
    # You may need to adapt this based on your specific chat format

    messages = conversation["messages"]
    prompt_text = ""

    for i, msg in enumerate(messages):
        role = msg["role"]
        content = msg["content"]

        # For assistant messages with structured content, just skip
        # (we only want to render up to where assistant should respond)
        if role == "assistant":
            # Add the role marker but not the content (model will generate)
            prompt_text += "Assistant:"
            break
        elif role == "user":
            # Add user message
            if isinstance(content, str):
                prompt_text += f"User: {content}\n"
            else:
                # If content is structured (list of parts), extract text
                text_parts = [p["text"] for p in content if p.get("type") == "text"]
                prompt_text += f"User: {''.join(text_parts)}\n"

    # Encode to tokens
    tokens = tokenizer.encode(prompt_text)
    return tokens


def simple_render_conversation(conversation: Dict, tokenizer) -> List[int]:
    """
    Simple conversation rendering that just takes the user's last message.

    This is a minimal approach that just uses the question as the prompt.

    Args:
        conversation: Conversation dict
        tokenizer: Tokenizer

    Returns:
        List of token IDs
    """
    messages = conversation["messages"]

    # Find the last user message
    for msg in messages:
        if msg["role"] == "user":
            content = msg["content"]
            if isinstance(content, str):
                return tokenizer.encode(content)
            else:
                # Extract text from structured content
                text_parts = [p["text"] for p in content if p.get("type") == "text"]
                return tokenizer.encode("".join(text_parts))

    # Fallback: empty prompt
    return []


def setup_gsm8k_task(evaluator, tokenizer, split: str = "test"):
    """
    Setup GSM8K task with the evaluator.

    Args:
        evaluator: ChatCoreEvaluator instance
        tokenizer: Tokenizer to use for encoding
        split: Dataset split ('train' or 'test')
    """
    from .gsm8k import evaluate_gsm8k, load_gsm8k_from_hf

    def load_fn(max_examples=None):
        """Load GSM8K data."""
        return load_gsm8k_from_hf(split=split, max_examples=max_examples)

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
