"""
ARC-Easy (AI2 Reasoning Challenge - Easy) evaluation task.
https://huggingface.co/datasets/allenai/ai2_arc

ARC is a multiple-choice question-answering dataset of genuine grade-school
level science questions. The dataset is partitioned into a Challenge set and
an Easy set, where the Challenge set contains questions answered incorrectly
by both a retrieval-based algorithm and a word co-occurrence algorithm.

Example:
    Question: Which of these is the best way to keep a cold drink cold?

    Choices:
    - Add ice to it = A
    - Keep it in a warm place = B
    - Store it in an insulated container = C
    - Leave it in direct sunlight = D

    Answer: C

The answer format is a single letter (A, B, C, D, etc.) corresponding to
the correct choice.
"""

from typing import Optional

from dataloaders.arc_dataloader import ARCDataLoader


def setup_arc_task(
    evaluator,
    tokenizer,
    subset: str = "ARC-Easy",
    split: str = "test",
    cache_dir: Optional[str] = "<YOUR_PATH>/divgoyal/nanochat_midtraining_data",
):
    """
    Setup ARC-Easy task with the evaluator.

    Args:
        evaluator: ChatCoreEvaluator instance
        tokenizer: Tokenizer to use for encoding
        subset: Dataset subset ('ARC-Easy' or 'ARC-Challenge')
        split: Dataset split ('train', 'validation', or 'test')
        cache_dir: Directory to cache the downloaded HuggingFace dataset
    """
    from .utils import render_conversation_for_completion

    # Initialize the dataloader
    dataloader = ARCDataLoader(subset=subset, split=split, cache_dir=cache_dir)

    def load_fn(max_examples=None):
        """Load ARC data using the dataloader."""
        examples = dataloader.load_data(max_examples=max_examples)
        # Add conversation formatting to each example
        for example in examples:
            example["conversation"] = dataloader.format_conversation(
                example["question"],
                example["choices"]["text"],
                example["choices"]["label"],
                example["answer"],
            )
        return examples

    def eval_fn(example, generated_text, return_details=False):
        """Evaluate an ARC prediction."""
        ground_truth_answer = example["answer"]
        is_correct = ARCDataLoader.evaluate(ground_truth_answer, generated_text)

        if return_details:
            # Extract predicted letter for comparison
            predicted_letter = (
                generated_text.strip()[0].upper() if generated_text.strip() else None
            )

            return {
                "success": is_correct,
                "reference_answer": ground_truth_answer,
                "predicted_answer": predicted_letter,
            }

        return is_correct

    def render_fn(example):
        """Render ARC conversation to prompt tokens."""
        conversation = example["conversation"]
        return render_conversation_for_completion(conversation, tokenizer)

    # Register with evaluator
    # Use "ARC-Easy" or "ARC-Challenge" as the task name
    task_name = subset.replace("-", "_").upper()  # e.g., "ARC_EASY" or "ARC_CHALLENGE"
    evaluator.register_task(
        task_name,
        {
            "load_fn": load_fn,
            "eval_fn": eval_fn,
            "render_fn": render_fn,
        },
    )
