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

from typing import Optional

from dataloaders.humaneval_dataloader import HumanEvalDataLoader


def setup_humaneval_task(
    evaluator,
    tokenizer,
    cache_dir: Optional[str] = "<YOUR_PATH>/divgoyal/nanochat_midtraining_data",
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

    # Initialize the dataloader
    dataloader = HumanEvalDataLoader(cache_dir=cache_dir, shuffle_seed=shuffle_seed)

    def load_fn(max_examples=None):
        """Load HumanEval data using the dataloader."""
        examples = dataloader.load_data(max_examples=max_examples)
        # Add conversation formatting to each example
        for example in examples:
            example["conversation"] = dataloader.format_conversation(
                example["prompt"], example["canonical_solution"]
            )
        return examples

    def eval_fn(example, generated_text, return_details=False):
        """Evaluate a HumanEval prediction."""
        return dataloader.evaluate(
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
