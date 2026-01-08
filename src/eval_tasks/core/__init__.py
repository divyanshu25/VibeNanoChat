"""
CORE evaluation benchmark implementation.

Based on the DCLM paper: https://arxiv.org/abs/2406.11794
"""

from eval_tasks.core.evaluator import CoreEvaluator
from eval_tasks.core.eval_tasks import evaluate_task, evaluate_example, forward_model
from eval_tasks.core.data_loading import load_core_tasks, load_jsonl
from eval_tasks.core.prompt_rendering import (
    render_prompts_mc,
    render_prompts_schema,
    render_prompts_lm,
)
from eval_tasks.core.token_batching import (
    batch_sequences_mc,
    batch_sequences_schema,
    batch_sequences_lm,
    find_common_length,
    stack_sequences,
)

__all__ = [
    # Main evaluator
    "CoreEvaluator",
    # Evaluation functions
    "evaluate_task",
    "evaluate_example",
    "forward_model",
    # Data loading
    "load_core_tasks",
    "load_jsonl",
    # Prompt rendering
    "render_prompts_mc",
    "render_prompts_schema",
    "render_prompts_lm",
    # Token batching
    "batch_sequences_mc",
    "batch_sequences_schema",
    "batch_sequences_lm",
    "find_common_length",
    "stack_sequences",
]
