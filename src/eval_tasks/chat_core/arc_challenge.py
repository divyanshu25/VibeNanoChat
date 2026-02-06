from typing import Optional

from dataloaders.arc_dataloader import ARCDataLoader


def setup_arc_challenge_task(
    evaluator,
    tokenizer,
    subset: str = "ARC-Challenge",
    split: str = "test",
    cache_dir: Optional[str] = "<YOUR_PATH>/divgoyal/nanochat_midtraining_data",
):
    """
    Setup ARC-Challenge task with the evaluator.
    """
    from .utils import render_conversation_for_completion

    # Initialize the dataloader
    dataloader = ARCDataLoader(subset=subset, split=split, cache_dir=cache_dir)

    def load_fn(max_examples=None):
        """Load ARC-Challenge data using the dataloader."""
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
        """Evaluate an ARC-Challenge prediction."""
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
        """Render ARC-Challenge conversation to prompt tokens."""
        conversation = example["conversation"]
        return render_conversation_for_completion(conversation, tokenizer)

    # Register with evaluator
    evaluator.register_task(
        "ARC_CHALLENGE",
        {
            "load_fn": load_fn,
            "eval_fn": eval_fn,
            "render_fn": render_fn,
        },
    )

    return evaluator
