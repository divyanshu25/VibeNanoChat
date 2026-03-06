"""
Utility functions for ChatCORE evaluation tasks.

Provides helper functions to:
- Render prompts from conversations for evaluation

Performance Note:
==================
The ChatCORE evaluator uses KV (Key-Value) caching for efficient generation:
- Without KV cache: Each token requires reprocessing all previous tokens → O(N²)
- With KV cache: Each token reuses cached attention keys/values → O(N)
- Result: 5-10x faster generation with ~2x memory usage

This is implemented transparently in the evaluator's generate methods.

Note:
=====
Dataset-specific loading and formatting functions have been moved to the
dataloaders package (src/dataloaders/) for better code reusability. See:
- ARCDataLoader for ARC datasets
- MMLUDataLoader for MMLU datasets
- HumanEvalDataLoader for HumanEval datasets
- GSM8KDataLoader for GSM8K datasets
"""

from typing import Dict, List


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


def render_conversation_for_training(conversation: Dict, tokenizer) -> tuple:
    """
    Render a conversation for training, returning token IDs and a mask.

    Creates a mask where 1 indicates tokens to train on (assistant responses)
    and 0 indicates tokens to ignore (special tokens, user messages).

    This is used for supervised fine-tuning (SFT) where we only want to
    compute loss on the assistant's responses, not the user's prompts.

    Args:
        conversation: Dict with 'messages' key containing list of messages.
                     Each message has 'role' and 'content' keys.
        tokenizer: Tokenizer with encode method (supports allowed_special="all")

    Returns:
        Tuple of (ids, mask) where:
        - ids: List of token IDs for the full conversation
        - mask: List of 0/1 indicating which tokens to train on

    Example:
        >>> conversation = {
        ...     'messages': [
        ...         {'role': 'user', 'content': 'What is 2+2?'},
        ...         {'role': 'assistant', 'content': '4'}
        ...     ]
        ... }
        >>> ids, mask = render_conversation_for_training(conversation, tokenizer)
        # ids: tokens for full conversation
        # mask: [0,0,0,...,1,1,1] (0 for user, 1 for assistant)
    """
    try:
        messages = conversation["messages"]
        formatted_parts = []
        mask_parts = []

        # Start with BOS token
        bos_text = "<|bos|>"
        bos_tokens = tokenizer.encode(bos_text, allowed_special="all")
        formatted_parts.extend(bos_tokens)
        mask_parts.extend([0] * len(bos_tokens))  # Don't train on BOS

        for msg in messages:
            role = msg["role"].lower()
            content = msg["content"]

            # Extract text content from structured or string format
            if isinstance(
                content, str
            ):  # if the content is a simple string, just add it to the text_content
                text_content = content
            elif isinstance(
                content, list
            ):  # if the content is a list of parts, we need to handle the parts
                # Handle structured content with parts (e.g., SpellingBee, GSM8K with tool calls)
                text_parts = []
                for part in content:
                    if part.get("type") == "text":
                        text_parts.append(part["text"])
                    elif part.get("type") == "python":
                        # Python tool call: <|python|>expression<|python_end|>
                        text_parts.append(f"<|python|>{part['text']}<|python_end|>")
                    elif part.get("type") == "output_start":
                        # Python output: <|output_start|>result<|output_end|>
                        text_parts.append(
                            f"<|output_start|>{part['text']}<|output_end|>"
                        )
                text_content = "".join(text_parts)
            else:  # if the content is not a string or a list of parts, convert it to a string
                text_content = str(content)

            if role == "user":
                # User message: <|user_start|>content<|user_end|>
                user_text = f"<|user_start|>{text_content}<|user_end|>"
                user_tokens = tokenizer.encode(user_text, allowed_special="all")
                formatted_parts.extend(user_tokens)
                mask_parts.extend([0] * len(user_tokens))  # Don't train on user tokens

            elif role == "assistant":
                # Assistant message: <|assistant_start|>content<|assistant_end|>
                # Train on content and end token, but not start token
                start_text = "<|assistant_start|>"
                start_tokens = tokenizer.encode(start_text, allowed_special="all")
                formatted_parts.extend(start_tokens)
                mask_parts.extend([0] * len(start_tokens))  # Don't train on start token

                # Content + end token (train on these)
                content_text = f"{text_content}<|assistant_end|>"
                content_tokens = tokenizer.encode(content_text, allowed_special="all")
                formatted_parts.extend(content_tokens)
                mask_parts.extend(
                    [1] * len(content_tokens)
                )  # Train on assistant content
            else:
                # Handle other roles (system, etc.) as user for consistency
                other_text = f"<|user_start|>{text_content}<|user_end|>"
                other_tokens = tokenizer.encode(other_text, allowed_special="all")
                formatted_parts.extend(other_tokens)
                mask_parts.extend([0] * len(other_tokens))  # Don't train on other roles

        return formatted_parts, mask_parts

    except Exception as e:
        print(f"Error in render_conversation_for_training: {e}")
        print(f"Conversation: {conversation}")
        import traceback

        traceback.print_exc()
        raise
