"""
GSM8K (Grade School Math 8K) evaluation task.
https://huggingface.co/datasets/openai/gsm8k

GSM8K is a dataset of 8.5K high quality grade school math word problems
that require multi-step reasoning. The dataset includes both the question
and a step-by-step solution with calculator tool calls.

Example:
    Question: Weng earns $12 an hour for babysitting. Yesterday, she just
              did 50 minutes of babysitting. How much did she earn?

    Answer: Weng earns 12/60 = $<<12/60=0.2>>0.2 per minute.
            Working 50 minutes, she earned 0.2 x 50 = $<<0.2*50=10>>10.
            #### 10

The answer format includes:
- Step-by-step reasoning with natural language
- Calculator tool calls in <<expression=result>> format
- Final answer after #### marker
"""

from typing import Optional

from dataloaders.gsm8k_dataloader import GSM8KDataLoader


def setup_gsm8k_task(
    evaluator,
    tokenizer,
    split: str = "test",
    cache_dir: Optional[str] = "/sensei-fs/users/divgoyal/nanochat_midtraining_data",
):
    """
    Setup GSM8K task with the evaluator.

    Args:
        evaluator: ChatCoreEvaluator instance
        tokenizer: Tokenizer to use for encoding
        split: Dataset split ('train' or 'test')
        cache_dir: Directory to cache the downloaded HuggingFace dataset
    """
    from .utils import render_conversation_for_completion

    # Initialize the dataloader
    dataloader = GSM8KDataLoader(split=split, cache_dir=cache_dir)

    def load_fn(max_examples=None):
        """Load GSM8K data using the dataloader."""
        examples = dataloader.load_data(max_examples=max_examples)
        # Add conversation formatting to each example
        for example in examples:
            example["conversation"] = dataloader.format_conversation(
                example["question"], example["answer"]
            )
        return examples

    def eval_fn(example, generated_text, return_details=False):
        """Evaluate a GSM8K prediction."""
        ground_truth_answer = example["answer"]
        is_correct = GSM8KDataLoader.evaluate(ground_truth_answer, generated_text)
        # if is_correct:
        #     print(f"{'='*80}")
        #     print(f"Reference answer: {ground_truth_answer}")
        #     print(f"{'-'*80}")
        #     print(f"Predicted answer: {generated_text}")
        #     print(f"{'='*80}")

        if return_details:
            # Extract answers for comparison
            ref_answer = GSM8KDataLoader.extract_answer(ground_truth_answer)
            pred_answer = GSM8KDataLoader.extract_answer(generated_text)

            return {
                "success": is_correct,
                "reference_answer": ref_answer,
                "predicted_answer": pred_answer,
            }

        return is_correct

    def render_fn(example):
        """Render GSM8K conversation to prompt tokens."""
        conversation = example["conversation"]
        return render_conversation_for_completion(conversation, tokenizer)

    # Register with evaluator
    evaluator.register_task(
        "GSM8K",
        {
            "load_fn": load_fn,
            "eval_fn": eval_fn,
            "render_fn": render_fn,
        },
    )
