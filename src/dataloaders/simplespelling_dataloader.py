"""
SimpleSpelling DataLoader

This module provides a data loader for the SimpleSpelling task: spelling words
letter by letter. This simpler task helps models learn character-level understanding
which is a prerequisite for more complex tasks like SpellingBee.

Example:
    >>> loader = SimpleSpellingDataLoader(size=100, split="train")
    >>> examples = loader.load_data()
    >>> print(f"Loaded {len(examples)} SimpleSpelling examples")
"""

import random
from typing import Dict, List, Optional

from dataloaders.word_utils import download_word_list

# Separate train and test with different random seeds
TEST_RANDOM_SEED_OFFSET = 10_000_000


class SimpleSpellingDataLoader:
    """
    A data loader for the SimpleSpelling task (spelling words).

    This loader generates synthetic examples where the model simply needs
    to spell out a word letter by letter. This is a simpler task designed
    to help models learn character-level understanding.

    Attributes:
        size (int): Number of examples to generate
        split (str): Dataset split ('train' or 'test')
        cache_dir (str): Directory to cache the word list
    """

    def __init__(
        self,
        size: int = 1000,
        split: str = "train",
        cache_dir: Optional[str] = None,
    ):
        """
        Initialize the SimpleSpelling data loader.

        Args:
            size: Number of examples to generate
            split: Dataset split to use ('train' or 'test')
            cache_dir: Directory to cache the word list
        """
        assert split in [
            "train",
            "test",
        ], "SimpleSpelling split must be 'train' or 'test'"

        self.size = size
        self.split = split
        self.cache_dir = cache_dir

        # Load word list
        words = download_word_list(cache_dir)

        # Shuffle with a different seed than SpellingBee
        rng = random.Random(42)
        rng.shuffle(words)
        self.words = words

    def load_data(self, max_examples: Optional[int] = None) -> List[Dict]:
        """
        Generate SimpleSpelling examples.

        Args:
            max_examples: Optional limit on number of examples (if None, uses self.size)

        Returns:
            List of examples, each with 'word', 'spelling', 'messages' keys
        """
        num_examples = (
            min(max_examples, self.size) if max_examples is not None else self.size
        )

        examples = []
        for index in range(num_examples):
            example = self._generate_example(index)
            examples.append(example)

        return examples

    def _generate_example(self, index: int) -> Dict:
        """
        Generate a single SimpleSpelling example.

        Args:
            index: Example index (used as random seed)

        Returns:
            Dictionary with example data
        """
        seed = index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)

        # Pick a random word
        word = rng.choice(self.words)

        # Spell it out
        word_letters = ",".join(list(word))

        # Create conversation
        messages = [
            {"role": "user", "content": f"Spell the word: {word}"},
            {"role": "assistant", "content": f"{word}:{word_letters}"},
        ]

        conversation = {
            "word": word,
            "spelling": word_letters,
            "messages": messages,
        }

        return conversation

    @staticmethod
    def evaluate(ground_truth_spelling: str, predicted_text: str) -> bool:
        """
        Evaluate a SimpleSpelling prediction against the ground truth.

        Args:
            ground_truth_spelling: The correct spelling (e.g., "h,e,l,l,o")
            predicted_text: The model's generated response text

        Returns:
            True if the predicted spelling matches ground truth, False otherwise
        """
        # Extract spelling from predicted text
        # Format is typically "word:h,e,l,l,o"
        if ":" in predicted_text:
            pred_spelling = predicted_text.split(":", 1)[1].strip()
        else:
            pred_spelling = predicted_text.strip()

        return pred_spelling == ground_truth_spelling


# --------SimpleSpelling DataLoader Testing--------#
if __name__ == "__main__":
    import json

    # Generate examples
    loader = SimpleSpellingDataLoader(size=5, split="train")
    examples = loader.load_data()

    # Pretty-print the JSON
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}")
        print(f"{'='*80}\n")
        print(json.dumps(example, indent=2, ensure_ascii=False))
        print()
