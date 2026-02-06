"""
Evaluation tasks for language model benchmarking.

Directory structure:
- core/: CORE benchmark implementation (DCLM paper)
- training/: Training-specific evaluation
- chat_core/: Chat model evaluation tasks
- utils/: Shared utilities
"""

# Main exports - Core evaluation (most commonly used)
from eval_tasks.core import CoreEvaluator
# Utils
from eval_tasks.utils import SimpleTokenizer

__all__ = [
    # Core evaluation (primary interface)
    "CoreEvaluator",
    # Utilities
    "SimpleTokenizer",
]
