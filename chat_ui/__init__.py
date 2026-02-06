"""
VibeNanoChat UI Package

A modular chat server for interacting with trained VibeNanoChat models.

Main components:
- ChatConfig: Configuration settings
- ModelManager: Model loading and inference
- SessionManager: Session and conversation management
- format_chat_prompt: Utility for formatting prompts
"""

from .config import ChatConfig
from .model_manager import ModelManager
from .prompt_utils import (format_chat_prompt, validate_conversation,
                           validate_message)
from .session_manager import SessionManager

__all__ = [
    "ChatConfig",
    "ModelManager",
    "SessionManager",
    "format_chat_prompt",
    "validate_message",
    "validate_conversation",
]

__version__ = "1.0.0"
