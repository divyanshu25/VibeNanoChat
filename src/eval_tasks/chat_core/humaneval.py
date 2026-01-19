"""
HumanEval evaluation task.
https://huggingface.co/datasets/openai/openai_humaneval

HumanEval is a dataset of 164 hand-written programming problems with function
signature, docstring, body, and tests. It is designed to evaluate functional
correctness of code generation models.

Example:
    Problem: Write a function to check if a given string is a palindrome.

    def is_palindrome(text: str) -> bool:
        '''
        Checks if given string is a palindrome
        >>> is_palindrome("racecar")
        True
        >>> is_palindrome("hello")
        False
        '''

    Model generates the function body, which is then tested against test cases.
"""

import re
from typing import Dict, List, Optional


def extract_imports(prompt: str) -> str:
    """
    Extract import statements from the beginning of a code block.

    Args:
        prompt: Python code that may contain import statements

    Returns:
        String containing all import statements found at the beginning

    Examples:
        >>> extract_imports("import math\\nfrom typing import List\\ndef foo(): pass")
        'import math\\nfrom typing import List'
        >>> extract_imports("def foo(): pass")
        ''
    """
    imports = []
    for line in prompt.split("\n"):
        stripped = line.strip()
        if stripped.startswith("import ") or stripped.startswith("from "):
            imports.append(stripped)
        elif stripped and not stripped.startswith("#"):
            # Stop at first non-import, non-comment line
            break
    return "\n".join(imports)


def extract_program(completion: str) -> str:
    """
    Extract Python code from LLM completion.

    Handles various output formats:
    - Code wrapped in ```python ... ``` or ``` ... ``` blocks
    - Plain code without markdown blocks
    - Extra text before/after code blocks

    Returns the first code block if found, otherwise returns the whole completion.

    Args:
        completion: The generated text from the model

    Returns:
        Extracted Python code

    Examples:
        >>> extract_program("```python\\nprint('hello')\\n```")
        "print('hello')"
        >>> extract_program("Here is code:\\n```\\nprint('hello')\\n```")
        "print('hello')"
        >>> extract_program("print('hello')")
        "print('hello')"
    """
    # Try to find markdown code blocks (```python or just ```)
    # Match ```python\n...\n``` or ```\n...\n```
    pattern = r"```(?:python)?\s*\n(.*?)\n```"
    matches = re.findall(pattern, completion, re.DOTALL)

    if matches:
        # Return the first code block found
        return matches[0].strip()

    # No code blocks found, return the whole completion
    return completion.strip()


def format_humaneval_conversation(prompt: str, canonical_solution: str) -> Dict:
    """
    Format a HumanEval example as a conversation.

    Args:
        prompt: The function signature and docstring
        canonical_solution: The correct implementation

    Returns:
        Conversation dict with 'messages' key containing user and assistant messages
        Example:
        {
            "messages": [
                {"role": "user", "content": "def add(a, b):\\n    '''Add two numbers'''\\n    "},
                {"role": "assistant", "content": "return a + b"},
            ]
        }
    """
    # The complete solution includes the prompt (signature) and the solution
    complete_solution = f"{prompt}\n{canonical_solution}"

    messages = [
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": complete_solution},
    ]

    return {"messages": messages}


def evaluate_humaneval(
    prompt: str,
    test: str,
    entry_point: str,
    predicted_text: str,
    timeout: float = 5.0,
    maximum_memory_bytes: Optional[int] = 256 * 1024 * 1024,
    return_details: bool = False,
):
    """
    Evaluate a HumanEval prediction by executing it against test cases.

    Args:
        prompt: The function signature and docstring (contains imports)
        test: Test cases to check the function
        entry_point: Name of the function to test
        predicted_text: The model's generated code
        timeout: Maximum execution time in seconds (default: 5.0)
        maximum_memory_bytes: Memory limit in bytes (default: 256MB)
        return_details: If True, return dict with execution details; if False, return bool

    Returns:
        If return_details=False: True if all tests pass, False otherwise
        If return_details=True: Dict with 'success', 'program', 'result', 'extracted_code'

    Examples:
        >>> prompt = "def add(a, b):\\n    '''Add two numbers'''\\n    "
        >>> test = "assert add(1, 2) == 3\\nassert add(-1, 1) == 0"
        >>> evaluate_humaneval(prompt, test, "add", "return a + b", timeout=5.0)
        True
    """
    try:
        from .execution import execute_code
    except ImportError:
        raise ImportError(
            "HumanEval evaluation requires the execution module. "
            "Make sure execution.py is available in the same directory."
        )

    # Extract imports from the prompt
    imports = extract_imports(prompt)

    # Extract the actual code from the completion (handles markdown blocks)
    completion_code = extract_program(predicted_text)

    # Construct the full program to execute
    program = (
        imports
        + "\n\n"
        + completion_code
        + "\n\n"
        + test
        + "\n"
        + f"check({entry_point})"
    )

    # Execute the code in a sandbox
    result = execute_code(
        program, timeout=timeout, maximum_memory_bytes=maximum_memory_bytes
    )
    # if result.success:
    #     print(f"{'='*80}")
    #     print(f"Program: {program}")
    #     print(f"{'-'*80}")
    #     print(f"Result: {result}")
    #     print(f"{'-'*80}")
    #     print(f"Extracted code: {completion_code}")
    #     print(f"{'-'*80}")
    #     print(f"{'='*80}")
    if return_details:
        return {
            "success": result.success,
            "program": program,
            "result": result,
            "extracted_code": completion_code,
        }

    return result.success


def load_humaneval_from_hf(
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = "/sensei-fs/users/divgoyal/nanochat_midtraining_data",
    shuffle_seed: int = 42,
) -> List[Dict]:
    """
    Load HumanEval dataset from HuggingFace.

    Args:
        max_examples: Optional limit on number of examples to load
        cache_dir: Directory to cache the downloaded dataset
        shuffle_seed: Random seed for shuffling (default: 42)

    Returns:
        List of examples, each with 'prompt', 'canonical_solution', 'test',
        'entry_point', and 'conversation' keys

    Note:
        Requires 'datasets' package: pip install datasets
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError(
            "Loading HumanEval requires the 'datasets' package. "
            "Install with: pip install datasets"
        )

    # Load from HuggingFace
    dataset = load_dataset("openai/openai_humaneval", split="test", cache_dir=cache_dir)

    # Shuffle with a fixed seed for reproducibility
    if shuffle_seed is not None:
        dataset = dataset.shuffle(seed=shuffle_seed)

    # Limit examples if requested
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    # Convert to our format
    examples = []
    for item in dataset:
        prompt = item["prompt"]
        canonical_solution = item["canonical_solution"]
        entry_point = item["entry_point"]
        test = item["test"]

        examples.append(
            {
                "prompt": prompt,
                "canonical_solution": canonical_solution,
                "test": test,
                "entry_point": entry_point,
                "conversation": format_humaneval_conversation(
                    prompt, canonical_solution
                ),
            }
        )

    return examples


def setup_humaneval_task(
    evaluator,
    tokenizer,
    cache_dir: Optional[str] = "/sensei-fs/users/divgoyal/nanochat_midtraining_data",
    shuffle_seed: int = 42,
):
    """
    Setup HumanEval task with the evaluator.

    Args:
        evaluator: ChatCoreEvaluator instance
        tokenizer: Tokenizer to use for encoding
        cache_dir: Directory to cache the downloaded HuggingFace dataset
        shuffle_seed: Random seed for shuffling the dataset
    """
    from .utils import render_conversation_for_completion

    def load_fn(max_examples=None):
        """Load HumanEval data."""
        return load_humaneval_from_hf(
            max_examples=max_examples, cache_dir=cache_dir, shuffle_seed=shuffle_seed
        )

    def eval_fn(example, generated_text, return_details=False):
        """Evaluate a HumanEval prediction."""
        return evaluate_humaneval(
            prompt=example["prompt"],
            test=example["test"],
            entry_point=example["entry_point"],
            predicted_text=generated_text,
            return_details=return_details,
        )

    def render_fn(example):
        """Render HumanEval conversation to prompt tokens."""
        conversation = example["conversation"]
        return render_conversation_for_completion(conversation, tokenizer)

    # Register with evaluator
    evaluator.register_task(
        "HumanEval",
        {
            "load_fn": load_fn,
            "eval_fn": eval_fn,
            "render_fn": render_fn,
        },
    )
