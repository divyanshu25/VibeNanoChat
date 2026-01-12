"""
Evaluation tasks for language model benchmarking.

Directory structure:
- core/: CORE benchmark implementation (DCLM paper)
- legacy/: Legacy evaluation implementations
- utils/: Shared utilities
"""

# Main exports - Core evaluation (most commonly used)
from eval_tasks.core import CoreEvaluator
# Legacy
from eval_tasks.legacy import HellaSwagDataloader
# Utils
from eval_tasks.utils import SimpleTokenizer

__all__ = [
    # Core evaluation (primary interface)
    "CoreEvaluator",
    # Utilities
    "SimpleTokenizer",
    # Legacy
    "HellaSwagDataloader",
]
