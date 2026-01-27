"""
GSM8K (Grade School Math 8K) DataLoader

This module provides a reusable data loader for the GSM8K dataset, which consists
of 8.5K high quality grade school math word problems that require multi-step reasoning.
The dataset includes both the question and a step-by-step solution with calculator
tool calls.

Dataset: https://huggingface.co/datasets/openai/gsm8k

Example:
    >>> loader = GSM8KDataLoader(split="test", cache_dir="/path/to/cache")
    >>> examples = loader.load_data(max_examples=100)
    >>> print(f"Loaded {len(examples)} GSM8K examples")
"""

import re
from typing import Dict, List, Optional

# Regex pattern to extract the numerical answer after #### marker
GSM_ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


class GSM8KDataLoader:
    """
    A reusable data loader for GSM8K (Grade School Math 8K) dataset.

    This loader provides clean access to the GSM8K math problems dataset
    with optional shuffling and limiting.

    Attributes:
        split (str): Dataset split ('train' or 'test')
        cache_dir (str): Directory to cache the downloaded dataset
        shuffle_seed (int): Random seed for shuffling
    """

    def __init__(
        self,
        split: str = "test",
        cache_dir: Optional[
            str
        ] = "/sensei-fs/users/divgoyal/nanochat_midtraining_data",
        shuffle_seed: int = 42,
    ):
        """
        Initialize the GSM8K data loader.

        Args:
            split: Dataset split to load ('train' or 'test')
            cache_dir: Directory to cache the downloaded dataset
            shuffle_seed: Random seed for shuffling (default: 42 for reproducibility)
        """
        self.split = split
        self.cache_dir = cache_dir
        self.shuffle_seed = shuffle_seed

    def load_data(
        self, max_examples: Optional[int] = None, format_as_conversation: bool = False
    ) -> List[Dict]:
        """
        Load GSM8K dataset from HuggingFace.

        Args:
            max_examples: Optional limit on number of examples to load
            format_as_conversation: If True, return examples in conversation format

        Returns:
            List of examples, each with 'question', 'answer', 'answer_value' keys
            (or conversation format if format_as_conversation=True)

        Note:
            Requires 'datasets' package: pip install datasets
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Loading GSM8K requires the 'datasets' package. "
                "Install with: pip install datasets"
            )

        # Load from HuggingFace with specified cache directory
        dataset = load_dataset(
            "openai/gsm8k", "main", split=self.split, cache_dir=self.cache_dir
        )

        # Shuffle for variety (deterministic with seed)
        dataset = dataset.shuffle(seed=self.shuffle_seed)

        # Convert to our format
        examples = []
        for i, item in enumerate(dataset):
            if max_examples is not None and i >= max_examples:
                break

            question = item["question"]
            answer = item["answer"]

            # Extract the numerical answer value
            answer_value = self.extract_answer(answer)

            examples.append(
                {
                    "question": question,
                    "answer": answer,
                    "answer_value": answer_value,
                    "raw_data": item,  # Include raw data for flexibility
                }
            )

        # Optionally convert to conversation format
        if format_as_conversation:
            conversations = []
            for example in examples:
                conv = self.format_conversation(example["question"], example["answer"])
                conversations.append(conv)
            return conversations

        return examples

    def format_conversation(self, question: str, answer: str) -> Dict:
        """
        Format a GSM8K example as a conversation.

        Args:
            question: The math problem question
            answer: The step-by-step answer with tool calls and final answer

        Returns:
            Conversation dict with 'messages' key containing user and assistant messages
        """
        # Parse the answer into structured parts
        answer_parts = self.parse_answer(answer)

        # Create conversation in chat format
        messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": answer_parts},
        ]

        return {"messages": messages}

    @staticmethod
    def extract_answer(text: str) -> Optional[str]:
        """
        Extract the numerical answer from a GSM8K response.

        Looks for the #### marker followed by a number. Normalizes by removing commas.

        Args:
            text: Response text that may contain #### marker with answer

        Returns:
            The extracted numerical answer as a string, or None if not found
        """
        match = GSM_ANSWER_RE.search(text)
        if match:
            answer_str = match.group(1).strip()
            # Remove commas from numbers (e.g., "1,234" -> "1234")
            answer_str = answer_str.replace(",", "")
            return answer_str
        return None

    @staticmethod
    def parse_answer(answer_text: str) -> List[Dict[str, str]]:
        """
        Parse GSM8K answer text into structured parts.

        GSM8K answers contain:
        - Regular text with reasoning
        - Calculator tool calls: <<expression=result>>
        - Final answer: #### number

        Args:
            answer_text: The raw answer string from GSM8K dataset

        Returns:
            List of parts, each a dict with 'type' and 'text' keys:
            - {'type': 'text', 'text': '...'} for regular text
            - {'type': 'python', 'text': 'expression'} for calculator input
            - {'type': 'output_start', 'text': 'result'} for calculator output
        """
        parts = []

        # Split on tool call markers: <<...>>
        segments = re.split(r"(<<[^>]+>>)", answer_text)

        for segment in segments:
            if segment.startswith("<<") and segment.endswith(">>"):
                # This is a calculator tool call
                inner = segment[2:-2]  # Remove << >>

                # Split on = to separate expression and result
                if "=" in inner:
                    expr, result = inner.rsplit("=", 1)
                    expr = expr.strip()
                    result = result.strip()
                else:
                    # No = sign, treat entire thing as expression
                    expr = inner.strip()
                    result = ""

                # Add expression as python input
                parts.append({"type": "python", "text": expr})
                # Add result as python output
                if result:
                    parts.append({"type": "output_start", "text": result})
            elif segment:
                # Regular text
                parts.append({"type": "text", "text": segment})

        return parts

    @staticmethod
    def evaluate(ground_truth_answer: str, predicted_text: str) -> bool:
        """
        Evaluate a GSM8K prediction against the ground truth.

        Extracts the numerical answer from both ground truth and prediction,
        then compares them for exact match.

        Args:
            ground_truth_answer: The full ground truth answer with #### marker
            predicted_text: The model's generated response text

        Returns:
            True if the predicted answer matches ground truth, False otherwise
        """
        # Extract numerical answers
        ref_answer = GSM8KDataLoader.extract_answer(ground_truth_answer)
        pred_answer = GSM8KDataLoader.extract_answer(predicted_text)

        # Both must have extractable answers to compare
        if ref_answer is None or pred_answer is None:
            return False

        # Compare as strings (already normalized by removing commas)
        return ref_answer == pred_answer


if __name__ == "__main__":
    import json

    # Generate examples
    loader = GSM8KDataLoader(split="train")
    examples = loader.load_data(max_examples=3)

    # Pretty-print the JSON
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}")
        print(f"{'='*80}\n")
        print(json.dumps(example, indent=2, ensure_ascii=False))
        print()
