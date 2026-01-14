"""
ChatCORE evaluation tasks for chat/generative models.

This module provides evaluation for chat-based tasks like:
- GSM8K: Math reasoning with step-by-step solutions
- MMLU: Multiple choice knowledge tasks (chat format)
- ARC: Science questions (chat format)
- HumanEval: Code generation
- SpellingBee: Spelling tasks

Unlike CORE (which uses likelihood-based evaluation), ChatCORE uses
generative evaluation where the model produces completions that are
then checked for correctness.
"""
