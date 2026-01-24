"""
MMLU (Massive Multitask Language Understanding) DataLoader

This module provides a reusable data loader for the MMLU dataset, which covers
57 subjects across STEM, humanities, social sciences, and more. It tests world
knowledge and problem-solving ability through multiple-choice questions with 4 options.

Dataset: https://huggingface.co/datasets/cais/mmlu

Example:
    >>> loader = MMLUDataLoader(
    ...     subset="all",
    ...     split="test",
    ...     cache_dir="/path/to/cache"
    ... )
    >>> examples = loader.load_data(max_examples=100)
    >>> print(f"Loaded {len(examples)} MMLU examples")
"""

from typing import Dict, List, Optional

# Standard MMLU answer letters
MMLU_LETTERS = ("A", "B", "C", "D")

# All 57 MMLU subjects
MMLU_SUBJECTS = (
    "abstract_algebra",
    "anatomy",
    "astronomy",
    "business_ethics",
    "clinical_knowledge",
    "college_biology",
    "college_chemistry",
    "college_computer_science",
    "college_mathematics",
    "college_medicine",
    "college_physics",
    "computer_security",
    "conceptual_physics",
    "econometrics",
    "electrical_engineering",
    "elementary_mathematics",
    "formal_logic",
    "global_facts",
    "high_school_biology",
    "high_school_chemistry",
    "high_school_computer_science",
    "high_school_european_history",
    "high_school_geography",
    "high_school_government_and_politics",
    "high_school_macroeconomics",
    "high_school_mathematics",
    "high_school_microeconomics",
    "high_school_physics",
    "high_school_psychology",
    "high_school_statistics",
    "high_school_us_history",
    "high_school_world_history",
    "human_aging",
    "human_sexuality",
    "international_law",
    "jurisprudence",
    "logical_fallacies",
    "machine_learning",
    "management",
    "marketing",
    "medical_genetics",
    "miscellaneous",
    "moral_disputes",
    "moral_scenarios",
    "nutrition",
    "philosophy",
    "prehistory",
    "professional_accounting",
    "professional_law",
    "professional_medicine",
    "professional_psychology",
    "public_relations",
    "security_studies",
    "sociology",
    "us_foreign_policy",
    "virology",
    "world_religions",
)


class MMLUDataLoader:
    """
    A reusable data loader for MMLU (Massive Multitask Language Understanding) dataset.

    This loader handles all 57 MMLU subjects or specific subsets, providing clean
    access to the dataset with optional shuffling and limiting.

    Attributes:
        subset (str): Dataset subset to load
        split (str): Dataset split ('train', 'validation', 'dev', or 'test')
        cache_dir (str): Directory to cache the downloaded dataset
        shuffle_seed (int): Random seed for shuffling
    """

    def __init__(
        self,
        subset: str = "all",
        split: str = "test",
        cache_dir: Optional[
            str
        ] = "/sensei-fs/users/divgoyal/nanochat_midtraining_data",
        shuffle_seed: int = 42,
    ):
        """
        Initialize the MMLU data loader.

        Args:
            subset: Dataset subset to load:
                    - "all" for all subjects combined
                    - "auxiliary_train" for the auxiliary training set
                    - Or a specific subject name (e.g., "college_biology")
            split: Dataset split to load:
                   - "test" for the test set (most common)
                   - "validation" for validation set
                   - "dev" for development set (5 examples per subject)
                   - "train" for training set (only with subset="auxiliary_train")
            cache_dir: Directory to cache the downloaded dataset
            shuffle_seed: Random seed for shuffling (default: 42 for reproducibility)

        Raises:
            AssertionError: If subset or split are invalid
        """
        # Validate inputs
        if subset != "all" and subset != "auxiliary_train":
            assert (
                subset in MMLU_SUBJECTS
            ), f"MMLU subset must be 'all', 'auxiliary_train', or one of {len(MMLU_SUBJECTS)} subjects, got: {subset}"

        if subset == "auxiliary_train":
            assert split == "train", "auxiliary_train subset only has 'train' split"
        else:
            assert split in [
                "validation",
                "dev",
                "test",
            ], f"MMLU split must be 'validation', 'dev', or 'test' (train only available with subset='auxiliary_train'), got: {split}"

        self.subset = subset
        self.split = split
        self.cache_dir = cache_dir
        self.shuffle_seed = shuffle_seed

    def load_data(
        self, max_examples: Optional[int] = None, format_as_conversation: bool = False
    ) -> List[Dict]:
        """
        Load MMLU dataset from HuggingFace.

        Args:
            max_examples: Optional limit on number of examples to load
            format_as_conversation: If True, return examples in conversation format

        Returns:
            List of examples, each with 'question', 'choices', 'answer', 'subject',
            'answer_letter' keys (or conversation format if format_as_conversation=True)

        Note:
            Requires 'datasets' package: pip install datasets
        """
        try:
            from datasets import load_dataset
        except ImportError:
            raise ImportError(
                "Loading MMLU requires the 'datasets' package. "
                "Install with: pip install datasets"
            )

        # Load from HuggingFace with specified cache directory
        dataset = load_dataset(
            "cais/mmlu", self.subset, split=self.split, cache_dir=self.cache_dir
        )

        # Handle auxiliary_train's nested structure
        if self.subset == "auxiliary_train":
            # The auxiliary_train rows have a weird additional 'train' wrapper
            dataset = dataset.map(lambda row: row["train"], remove_columns=["train"])

        # Shuffle for variety (deterministic with seed)
        dataset = dataset.shuffle(seed=self.shuffle_seed)

        # Limit examples if requested
        if max_examples is not None:
            dataset = dataset.select(range(min(max_examples, len(dataset))))

        # Convert to our format
        examples = []
        for item in dataset:
            question = item["question"]
            choices = item["choices"]
            answer = item["answer"]  # This is an integer index (0-3)
            subject = item["subject"]

            # Validate
            assert len(choices) == 4, f"MMLU should have 4 choices, got {len(choices)}"
            assert 0 <= answer < 4, f"MMLU answer should be 0-3, got {answer}"

            examples.append(
                {
                    "question": question,
                    "choices": choices,
                    "answer": answer,
                    "answer_letter": MMLU_LETTERS[answer],
                    "subject": subject,
                    "raw_data": item,  # Include raw data for flexibility
                }
            )

        # Optionally convert to conversation format
        if format_as_conversation:
            conversations = []
            for example in examples:
                conv = self.format_conversation(
                    example["question"],
                    example["choices"],
                    example["answer"],
                    example["subject"],
                )
                conversations.append(conv)
            return conversations

        return examples

    def format_conversation(
        self, question: str, choices: List[str], answer: int, subject: str
    ) -> Dict:
        """
        Format an MMLU example as a conversation.

        Args:
            question: The question text
            choices: List of 4 choice texts
            answer: Index of the correct answer (0-3, corresponding to A-D)
            subject: The subject category (e.g., "college_biology")

        Returns:
            Conversation dict with 'messages', 'letters', and 'subject' keys
        """
        # Validate inputs
        assert len(choices) == 4, f"MMLU should have 4 choices, got {len(choices)}"
        assert 0 <= answer < 4, f"MMLU answer index should be 0-3, got {answer}"

        # Render the question in multiple-choice format
        query = f"Multiple Choice question: {question}\n"
        query += "".join(
            [f"{letter}. {choice}\n" for letter, choice in zip(MMLU_LETTERS, choices)]
        )
        query += "\nOnly one choice is correct. Start your response with the letter of the correct answer."

        # Get the answer letter
        answer_letter = MMLU_LETTERS[answer]

        messages = [
            {"role": "user", "content": query},
            {"role": "assistant", "content": answer_letter},
        ]

        conversation = {
            "messages": messages,
            "letters": MMLU_LETTERS,  # Useful for evaluation
            "subject": subject,  # Useful for per-subject metrics
        }

        return conversation

    @staticmethod
    def evaluate(ground_truth_answer: str, predicted_text: str) -> bool:
        """
        Evaluate an MMLU prediction against the ground truth.

        Args:
            ground_truth_answer: The correct answer letter (e.g., "A", "B", "C", "D")
            predicted_text: The model's generated response text

        Returns:
            True if the predicted answer matches ground truth, False otherwise
        """
        # Extract the predicted letter
        predicted_text = predicted_text.strip()
        if not predicted_text:
            return False

        # Compare (case-insensitive)
        return predicted_text[0].upper() == ground_truth_answer.upper()


if __name__ == "__main__":
    import json

    # Generate examples
    loader = MMLUDataLoader(subset="abstract_algebra", split="test")
    examples = loader.load_data(max_examples=3)

    # Pretty-print the JSON
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}")
        print(f"{'='*80}\n")
        print(json.dumps(example, indent=2, ensure_ascii=False))
        print()
