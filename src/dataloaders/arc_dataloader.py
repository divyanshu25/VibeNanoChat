"""
ARC (AI2 Reasoning Challenge) DataLoader

This module provides a reusable data loader for the ARC dataset, which includes
both ARC-Easy and ARC-Challenge subsets. ARC is a multiple-choice question-answering
dataset of genuine grade-school level science questions.

Dataset: https://huggingface.co/datasets/allenai/ai2_arc

Example:
    >>> loader = ARCDataLoader(
    ...     subset="ARC-Easy",
    ...     split="test",
    ...     cache_dir="/path/to/cache"
    ... )
    >>> examples = loader.load_data(max_examples=100)
    >>> print(f"Loaded {len(examples)} ARC examples")
"""

from typing import Dict, List, Optional


class ARCDataLoader:
    """
    A reusable data loader for ARC (AI2 Reasoning Challenge) dataset.

    This loader handles both ARC-Easy and ARC-Challenge subsets, providing
    clean access to the dataset with optional shuffling and limiting.

    Attributes:
        subset (str): Dataset subset ('ARC-Easy' or 'ARC-Challenge')
        split (str): Dataset split ('train', 'validation', or 'test')
        cache_dir (str): Directory to cache the downloaded dataset
        shuffle_seed (int): Random seed for shuffling
    """

    def __init__(
        self,
        subset: str = "ARC-Easy",
        split: str = "test",
        cache_dir: Optional[
            str
        ] = "<YOUR_PATH>/divgoyal/nanochat_midtraining_data",
        shuffle_seed: int = 42,
    ):
        """
        Initialize the ARC data loader.

        Args:
            subset: Dataset subset ('ARC-Easy' or 'ARC-Challenge')
            split: Dataset split to load ('train', 'validation', or 'test')
            cache_dir: Directory to cache the downloaded dataset
            shuffle_seed: Random seed for shuffling (default: 42 for reproducibility)

        Raises:
            AssertionError: If subset or split are invalid
        """
        # Validate inputs
        assert subset in [
            "ARC-Easy",
            "ARC-Challenge",
        ], f"ARC subset must be 'ARC-Easy' or 'ARC-Challenge', got: {subset}"
        assert split in [
            "train",
            "validation",
            "test",
        ], f"ARC split must be 'train', 'validation', or 'test', got: {split}"

        self.subset = subset
        self.split = split
        self.cache_dir = cache_dir
        self.shuffle_seed = shuffle_seed

    def load_data(
        self, max_examples: Optional[int] = None, format_as_conversation: bool = False
    ) -> List[Dict]:
        """
        Load ARC dataset from HuggingFace.

        Args:
            max_examples: Optional limit on number of examples to load
            format_as_conversation: If True, return examples in conversation format

        Returns:
            List of examples, each with 'question', 'answer', 'choices' keys
            (or conversation format if format_as_conversation=True)

        Note:
            Requires 'datasets' package: pip install datasets
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Loading ARC requires the 'datasets' package. "
                "Install with: pip install datasets"
            )

        # Load from HuggingFace with specified cache directory
        dataset = load_dataset(
            "allenai/ai2_arc", self.subset, split=self.split, cache_dir=self.cache_dir
        )

        # Shuffle for variety (deterministic with seed)
        dataset = dataset.shuffle(seed=self.shuffle_seed)

        # Limit examples if requested
        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        # Convert to our format
        examples = []
        for item in dataset:
            question = item["question"]
            choices_text = item["choices"]["text"]
            choices_labels = item["choices"]["label"]
            answer_key = item["answerKey"]

            # Sanity check
            assert (
                answer_key in choices_labels
            ), f"ARC answer {answer_key} must be one of {choices_labels}"

            examples.append(
                {
                    "question": question,
                    "answer": answer_key,
                    "choices": {
                        "text": choices_text,
                        "label": choices_labels,
                    },
                    "raw_data": item,  # Include raw data for flexibility
                }
            )

        # Optionally convert to conversation format
        if format_as_conversation:
            conversations = []
            for example in examples:
                conv = self.format_conversation(
                    example["question"],
                    example["choices"]["text"],
                    example["choices"]["label"],
                    example["answer"],
                )
                conversations.append(conv)
            return conversations

        return examples

    def format_conversation(
        self,
        question: str,
        choices_text: List[str],
        choices_labels: List[str],
        answer_key: str,
    ) -> Dict:
        """
        Format an ARC example as a conversation.

        Args:
            question: The question text
            choices_text: List of choice texts
            choices_labels: List of choice labels (e.g., ["A", "B", "C", "D"])
            answer_key: The correct answer letter (e.g., "A")

        Returns:
            Conversation dict with 'messages' and 'letters' keys
        """
        # Render the question in multiple-choice format
        query = f"Multiple Choice question: {question}\n"
        query += "".join(
            [
                f"{label}. {choice}\n"
                for label, choice in zip(choices_labels, choices_text)
            ]
        )
        query += "\nOnly one choice is correct. Start your response with the letter of the correct answer."

        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer_key},
        ]

        conversation = {
            "messages": messages,
            "letters": choices_labels,  # Useful for evaluation and letter constraint
        }
        return conversation

    @staticmethod
    def evaluate(ground_truth_letter: str, predicted_text: str) -> bool:
        """
        Evaluate an ARC prediction against the ground truth.

        Args:
            ground_truth_letter: The correct answer letter (e.g., "A")
            predicted_text: The model's generated response text

        Returns:
            True if the predicted answer matches ground truth, False otherwise
        """
        if not predicted_text:
            return False

        # Extract the first character (should be the letter)
        predicted_text = predicted_text.strip()
        if not predicted_text:
            return False

        predicted_letter = predicted_text[0].upper()

        return predicted_letter == ground_truth_letter.upper()


if __name__ == "__main__":
    import json

    # Generate examples
    loader = ARCDataLoader(subset="ARC-Easy", split="validation")
    examples = loader.load_data(max_examples=3)

    # Pretty-print the JSON
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}")
        print(f"{'='*80}\n")
        print(json.dumps(example, indent=2, ensure_ascii=False))
        print()
