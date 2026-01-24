"""
HumanEval DataLoader

This module provides a reusable data loader for the HumanEval dataset, which is
a set of 164 hand-written programming problems with function signature, docstring,
body, and tests. It is designed to evaluate functional correctness of code generation.

Dataset: https://huggingface.co/datasets/openai/openai_humaneval

Example:
    >>> loader = HumanEvalDataLoader(cache_dir="/path/to/cache")
    >>> examples = loader.load_data(max_examples=50)
    >>> print(f"Loaded {len(examples)} HumanEval examples")
"""

import re
from typing import Dict, List, Optional


class HumanEvalDataLoader:
    """
    A reusable data loader for HumanEval dataset.

    This loader provides clean access to the HumanEval programming problems
    dataset with optional shuffling and limiting.

    Attributes:
        cache_dir (str): Directory to cache the downloaded dataset
        shuffle_seed (int): Random seed for shuffling
    """

    def __init__(
        self,
        cache_dir: Optional[
            str
        ] = "/sensei-fs/users/divgoyal/nanochat_midtraining_data",
        shuffle_seed: int = 42,
    ):
        """
        Initialize the HumanEval data loader.

        Args:
            cache_dir: Directory to cache the downloaded dataset
            shuffle_seed: Random seed for shuffling (default: 42 for reproducibility)
        """
        self.cache_dir = cache_dir
        self.shuffle_seed = shuffle_seed

    def load_data(self, max_examples: Optional[int] = None) -> List[Dict]:
        """
        Load HumanEval dataset from HuggingFace.

        Args:
            max_examples: Optional limit on number of examples to load

        Returns:
            List of examples, each with 'prompt', 'canonical_solution', 'test',
            'entry_point', 'task_id' keys

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

        # Load from HuggingFace (HumanEval only has 'test' split)
        dataset = load_dataset(
            "openai/openai_humaneval", split="test", cache_dir=self.cache_dir
        )

        # Shuffle with a fixed seed for reproducibility
        if self.shuffle_seed is not None:
            dataset = dataset.shuffle(seed=self.shuffle_seed)

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
            task_id = item.get("task_id", "")

            examples.append(
                {
                    "prompt": prompt,
                    "canonical_solution": canonical_solution,
                    "test": test,
                    "entry_point": entry_point,
                    "task_id": task_id,
                    "raw_data": item,  # Include raw data for flexibility
                }
            )

        return examples

    def format_conversation(self, prompt: str, canonical_solution: str) -> Dict:
        """
        Format a HumanEval example as a conversation.

        Args:
            prompt: The function signature and docstring
            canonical_solution: The correct implementation

        Returns:
            Conversation dict with 'messages' key containing user and assistant messages
        """
        # Add instruction to make the task clear
        user_message = f"Complete the following Python function:\n\n{prompt}"

        messages = [
            {"role": "user", "content": user_message},
            {
                "role": "assistant",
                "content": canonical_solution,
            },  # Only the solution, not the prompt
        ]

        return {"messages": messages}

    @staticmethod
    def extract_imports(prompt: str) -> str:
        """
        Extract import statements from the beginning of a code block.

        Args:
            prompt: Python code that may contain import statements

        Returns:
            String containing all import statements found at the beginning
        """
        imports = []
        for line in prompt.split("\n"):
            stripped = line.strip()  # this removes whitespace and comments
            if stripped.startswith("import ") or stripped.startswith(
                "from "
            ):  # this checks if the line starts with import or from
                imports.append(stripped)
            elif stripped and not stripped.startswith("#"):
                # Stop at first non-import, non-comment line
                break
        return "\n".join(imports)

    @staticmethod
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

    def evaluate(
        self,
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

        Note:
            Requires the execution module for sandboxed code execution
        """
        try:
            # Import execution module (needs to be available in the project)
            from eval_tasks.chat_core.execution import execute_code
        except ImportError:
            raise ImportError(
                "HumanEval evaluation requires the execution module. "
                "Make sure execution.py is available in eval_tasks/chat_core/"
            )

        # Extract imports from the prompt
        imports = self.extract_imports(prompt)

        # Extract the actual code from the completion (handles markdown blocks)
        completion_code = self.extract_program(predicted_text)

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

        # print("PROGRAM (sent to execute_code):")
        # print("=" * 80)
        # print(program)
        # print("=" * 80)

        # Execute the code in a sandbox
        result = execute_code(
            program, timeout=timeout, maximum_memory_bytes=maximum_memory_bytes
        )

        if return_details:
            return {
                "success": result.success,
                "program": program,
                "result": result,
                "extracted_code": completion_code,
            }

        return result.success


if __name__ == "__main__":
    import json

    # Generate examples
    loader = HumanEvalDataLoader()
    examples = loader.load_data(max_examples=3)

    # Pretty-print the JSON
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}")
        print(f"{'='*80}\n")
        print(json.dumps(example, indent=2, ensure_ascii=False))
        print()
