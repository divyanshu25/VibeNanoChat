"""
MMLU (Massive Multitask Language Understanding) evaluation task.
https://huggingface.co/datasets/cais/mmlu

MMLU is a benchmark covering 57 subjects across STEM, humanities, social sciences,
and more. It tests world knowledge and problem-solving ability through multiple-choice
questions with 4 options each (A, B, C, D).

Example:
    Question: What is the capital of France?
    Choices:
        A. London
        B. Berlin
        C. Paris
        D. Rome
    Answer: C

The dataset includes:
- Multiple subject categories (math, history, law, medicine, etc.)
- 4-choice multiple-choice format
- Questions requiring various levels of knowledge and reasoning
"""

from typing import Optional

from dataloaders.mmlu_dataloader import MMLUDataLoader

from .utils import render_conversation_for_completion


def setup_mmlu_task(
    evaluator,
    tokenizer,
    subset: str = "all",
    split: str = "test",
    cache_dir: Optional[str] = "<YOUR_PATH>/divgoyal/nanochat_midtraining_data",
):
    """
    Setup MMLU task with the evaluator.

    Args:
        evaluator: ChatCoreEvaluator instance
        tokenizer: Tokenizer to use for encoding
        subset: Dataset subset to load (default: "all" for all subjects)
        split: Dataset split (default: "test")
        cache_dir: Directory to cache the downloaded HuggingFace dataset
    """

    # Initialize the dataloader
    dataloader = MMLUDataLoader(subset=subset, split=split, cache_dir=cache_dir)

    def load_fn(max_examples=None):
        """Load MMLU data using the dataloader."""
        examples = dataloader.load_data(max_examples=max_examples)
        # Add conversation formatting to each example
        for example in examples:
            example["conversation"] = dataloader.format_conversation(
                example["question"],
                example["choices"],
                example["answer"],
                example["subject"],
            )
        return examples

    def eval_fn(example, generated_text, return_details=False):
        """Evaluate an MMLU prediction."""
        ground_truth_letter = example["answer_letter"]
        is_correct = MMLUDataLoader.evaluate(ground_truth_letter, generated_text)

        if return_details:
            # Extract prediction for comparison
            pred_letter = (
                generated_text.strip()[0].upper() if generated_text.strip() else None
            )

            return {
                "success": is_correct,
                "reference_answer": ground_truth_letter,
                "predicted_answer": pred_letter,
                "subject": example.get("subject", "unknown"),
            }

        return is_correct

    def render_fn(example):
        """Render MMLU conversation to prompt tokens."""
        conversation = example["conversation"]
        return render_conversation_for_completion(conversation, tokenizer)

    # Determine task name based on subset
    if subset == "all":
        task_name = "MMLU"
    elif subset == "auxiliary_train":
        task_name = "MMLU-AuxTrain"
    else:
        # Capitalize subject name for display
        task_name = f"MMLU-{subset.replace('_', ' ').title()}"

    # Register with evaluator
    evaluator.register_task(
        task_name,
        {
            "load_fn": load_fn,
            "eval_fn": eval_fn,
            "render_fn": render_fn,
        },
    )
