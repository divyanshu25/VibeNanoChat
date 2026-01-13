"""
Tool use support for ChatCORE evaluation.

Provides calculator tool execution functionality that models can use during generation.
This allows models to perform accurate mathematical computations rather than
hallucinating numeric results.

Based on the tool-use implementation from nanochat/engine.py.
"""

import signal
import warnings
from contextlib import contextmanager
from typing import Optional

# -----------------------------------------------------------------------------
# Safe execution helpers
# -----------------------------------------------------------------------------


@contextmanager
def timeout(duration, formula):
    """
    Context manager to limit execution time of a code block.

    Args:
        duration: Maximum seconds to allow
        formula: Formula being evaluated (for error messages)

    Raises:
        Exception: If execution exceeds duration
    """

    def timeout_handler(signum, frame):
        raise Exception(f"'{formula}': timed out after {duration} seconds")

    signal.signal(signal.SIGALRM, timeout_handler)
    signal.alarm(duration)
    yield
    signal.alarm(0)


def eval_with_timeout(formula, max_time=3):
    """
    Safely evaluate a Python expression with timeout protection.

    Args:
        formula: String containing Python expression to evaluate
        max_time: Maximum seconds to allow for evaluation

    Returns:
        Result of evaluation, or None if failed/timed out
    """
    try:
        with timeout(max_time, formula):
            with warnings.catch_warnings():
                warnings.simplefilter("ignore", SyntaxWarning)
                return eval(formula, {"__builtins__": {}}, {})
    except Exception:
        signal.alarm(0)
        return None


# -----------------------------------------------------------------------------
# Calculator tool
# -----------------------------------------------------------------------------


def use_calculator(expr: str) -> Optional[float]:
    """
    Evaluate a Python expression safely as a calculator tool.

    Supports:
    - Basic math: +, -, *, /, parentheses
    - String operations: .count() method

    Safety features:
    - Sandboxed execution (no builtins)
    - Timeout protection (3 seconds)
    - Whitelist-based validation
    - Blocks dangerous patterns

    Args:
        expr: String expression to evaluate (e.g., "12/60" or "'hello'.count('l')")

    Returns:
        Evaluation result as a number/string, or None if invalid/failed

    Examples:
        >>> use_calculator("12/60")
        0.2
        >>> use_calculator("0.2 * 50")
        10.0
        >>> use_calculator("'strawberry'.count('r')")
        3
        >>> use_calculator("2**10")  # Power operator blocked
        None
    """
    # Remove commas from numbers (e.g., "1,234" -> "1234")
    expr = expr.replace(",", "")

    # Check if it's a pure math expression
    if all([x in "0123456789*+-/.() " for x in expr]):
        if "**" in expr:  # Disallow power operator
            return None
        return eval_with_timeout(expr)

    # Check if it's a string operation we support
    # Allow: strings (single/double quotes), .count(), letters, numbers, spaces, parens
    allowed_chars = (
        "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'\"()._ "
    )
    if not all([x in allowed_chars for x in expr]):
        return None

    # Disallow dangerous patterns
    dangerous_patterns = [
        "__",
        "import",
        "exec",
        "eval",
        "compile",
        "open",
        "file",
        "input",
        "raw_input",
        "globals",
        "locals",
        "vars",
        "dir",
        "getattr",
        "setattr",
        "delattr",
        "hasattr",
    ]
    expr_lower = expr.lower()
    if any(pattern in expr_lower for pattern in dangerous_patterns):
        return None

    # Only allow .count() method for now (can expand later)
    if ".count(" not in expr:
        return None

    # Evaluate with timeout
    return eval_with_timeout(expr)
