"""
SmolTalk DataLoader

This module provides a reusable data loader for the SmolTalk dataset, which is a
large-scale conversational dataset containing various types of interactions including
general chat, reasoning, and instruction-following conversations.

Dataset: https://huggingface.co/datasets/HuggingFaceTB/smoltalk

Example:
    >>> loader = SmolTalkDataLoader(
    ...     config="all",
    ...     split="train",
    ...     cache_dir="/path/to/cache"
    ... )
    >>> examples = loader.load_data(max_examples=1000)
    >>> print(f"Loaded {len(examples)} SmolTalk examples")
"""

from typing import Dict, List, Optional


class SmolTalkDataLoader:
    """
    A reusable data loader for SmolTalk conversational dataset.

    This loader provides clean access to the SmolTalk dataset with optional
    shuffling, sampling, and limiting.

    Attributes:
        config (str): Dataset config to load (e.g., "all", "smol-magpie-ultra", etc.)
        split (str): Dataset split ('train' or 'test')
        cache_dir (str): Directory to cache the downloaded dataset
        shuffle_seed (int): Random seed for shuffling
    """

    def __init__(
        self,
        config: str = "all",
        split: str = "train",
        cache_dir: Optional[
            str
        ] = "<YOUR_PATH>/divgoyal/nanochat_midtraining_data",
        shuffle_seed: int = 42,
    ):
        """
        Initialize the SmolTalk data loader.

        Args:
            config: Dataset config to load. Options include:
                   - "all" for all conversation types combined
                   - "smol-magpie-ultra" for instruction-following
                   - "smol-rewrite" for rewrite tasks
                   - "smol-constraints" for constrained generation
                   And others. See HuggingFace dataset page for full list.
            split: Dataset split to load ('train' or 'test')
            cache_dir: Directory to cache the downloaded dataset
            shuffle_seed: Random seed for shuffling (default: 42 for reproducibility)
        """
        self.config = config
        self.split = split
        self.cache_dir = cache_dir
        self.shuffle_seed = shuffle_seed

    def load_data(
        self, max_examples: Optional[int] = None, sample_first: bool = True
    ) -> List[Dict]:
        """
        Load SmolTalk dataset from HuggingFace.

        Args:
            max_examples: Optional limit on number of examples to load.
                         If the dataset is larger than this, it will be sampled.
            sample_first: If True and max_examples is set, randomly sample the data
                         before processing. If False, just take the first N examples.

        Returns:
            List of examples, each with 'messages' key containing the conversation
            and 'raw_data' key with the original dataset item.

        Note:
            Requires 'datasets' package: pip install datasets
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Loading SmolTalk requires the 'datasets' package. "
                "Install with: pip install datasets"
            )

        # Load from HuggingFace with specified cache directory
        dataset = load_dataset(
            "HuggingFaceTB/smoltalk",
            self.config,
            split=self.split,
            cache_dir=self.cache_dir,
        )

        # Shuffle for variety (deterministic with seed)
        if self.shuffle_seed is not None:
            dataset = dataset.shuffle(seed=self.shuffle_seed)

        # Sample/limit examples if requested
        if max_examples is not None and len(dataset) > max_examples:
            if sample_first:
                # Already shuffled above, so just take first N
                dataset = dataset.select(range(max_examples))
            else:
                # Take first N without shuffling first
                dataset = dataset.select(range(min(max_examples, len(dataset))))

        # Convert to our format
        examples = []
        for item in dataset:
            messages = item["messages"]

            # Validate message structure
            if not isinstance(messages, list):
                continue

            # Ensure all messages have role and content
            valid = True
            for msg in messages:
                if (
                    not isinstance(msg, dict)
                    or "role" not in msg
                    or "content" not in msg
                ):
                    valid = False
                    break

            if not valid:
                continue

            # SmolTalk data is already in the correct format
            # format_as_conversation parameter kept for API consistency
            examples.append(
                {
                    "messages": messages,
                    "raw_data": item,  # Include raw data for flexibility
                }
            )

        return examples

    @staticmethod
    def validate_conversation(messages: List[Dict]) -> bool:
        """
        Validate that a conversation has the correct structure.

        Args:
            messages: List of message dicts to validate

        Returns:
            True if valid, False otherwise
        """
        if not isinstance(messages, list) or len(messages) == 0:
            return False

        for msg in messages:
            if not isinstance(msg, dict):
                return False
            if "role" not in msg or "content" not in msg:
                return False
            if msg["role"] not in ["user", "assistant", "system"]:
                return False

        return True

    def get_statistics(self, examples: List[Dict]) -> Dict:
        """
        Get statistics about the loaded examples.

        Args:
            examples: List of examples from load_data()

        Returns:
            Dict with statistics like message counts, role distribution, etc.
        """
        total_messages = 0
        role_counts = {"user": 0, "assistant": 0, "system": 0, "other": 0}
        turn_counts = []

        for example in examples:
            messages = example["messages"]
            total_messages += len(messages)
            turn_counts.append(len(messages))

            for msg in messages:
                role = msg["role"].lower()
                if role in role_counts:
                    role_counts[role] += 1
                else:
                    role_counts["other"] += 1

        avg_turns = sum(turn_counts) / len(turn_counts) if turn_counts else 0

        return {
            "num_conversations": len(examples),
            "total_messages": total_messages,
            "avg_turns_per_conversation": avg_turns,
            "min_turns": min(turn_counts) if turn_counts else 0,
            "max_turns": max(turn_counts) if turn_counts else 0,
            "role_distribution": role_counts,
        }


if __name__ == "__main__":
    import json

    # Generate examples
    loader = SmolTalkDataLoader(config="all", split="train")
    examples = loader.load_data(max_examples=3)

    # Pretty-print the JSON
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}")
        print(f"{'='*80}\n")
        print(json.dumps(example, indent=2, ensure_ascii=False))
        print()
