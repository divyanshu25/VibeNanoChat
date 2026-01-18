"""
Prompt formatting utilities for the NanoGPT Chat Server.

Handles formatting conversation histories into prompts.
"""

from typing import Dict, List

from .config import ChatConfig


def format_chat_prompt(messages: List[Dict[str, str]]) -> str:
    """
    Format conversation history into the chat prompt format.

    Args:
        messages: List of message dicts with 'role' and 'content' keys.
                 Role should be either 'user' or 'assistant'.

    Returns:
        The formatted prompt string ready for model input.

    Example:
        >>> messages = [
        ...     {"role": "user", "content": "Hello!"},
        ...     {"role": "assistant", "content": "Hi there!"},
        ...     {"role": "user", "content": "How are you?"}
        ... ]
        >>> prompt = format_chat_prompt(messages)
    """
    tokens = ChatConfig.CHAT_TOKENS

    prompt = tokens["bos"]
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"{tokens['user_start']}{msg['content']}{tokens['user_end']}"
        elif msg["role"] == "assistant":
            prompt += (
                f"{tokens['assistant_start']}{msg['content']}{tokens['assistant_end']}"
            )

    # Add the assistant start token to prompt the model to respond
    prompt += tokens["assistant_start"]

    return prompt


def validate_message(message: dict) -> bool:
    """
    Validate that a message has the correct format.

    Args:
        message: A message dict to validate.

    Returns:
        True if the message is valid, False otherwise.
    """
    if not isinstance(message, dict):
        return False

    if "role" not in message or "content" not in message:
        return False

    if message["role"] not in ["user", "assistant"]:
        return False

    if not isinstance(message["content"], str):
        return False

    return True


def validate_conversation(conversation: List[dict]) -> bool:
    """
    Validate that a conversation history has the correct format.

    Args:
        conversation: List of message dicts to validate.

    Returns:
        True if all messages are valid, False otherwise.
    """
    if not isinstance(conversation, list):
        return False

    return all(validate_message(msg) for msg in conversation)
