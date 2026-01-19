"""
ChatCORE evaluation tasks for chat/generative models.

This module provides evaluation for chat-based tasks like:
- GSM8K: Math reasoning with step-by-step solutions
- HumanEval: Code generation with test execution
- MMLU: Multiple choice knowledge tasks (chat format)
- ARC: Science questions (chat format)
- SpellingBee: Spelling tasks

Unlike CORE (which uses likelihood-based evaluation), ChatCORE uses
generative evaluation where the model produces completions that are
then checked for correctness.
"""

from .arc_challenge import setup_arc_challenge_task
from .arc_easy import setup_arc_task
from .evaluator import ChatCoreEvaluator
from .gsm8k import setup_gsm8k_task
from .humaneval import setup_humaneval_task

__all__ = [
    "ChatCoreEvaluator",
    "setup_gsm8k_task",
    "setup_humaneval_task",
    "setup_arc_task",
    "setup_arc_challenge_task",
]
