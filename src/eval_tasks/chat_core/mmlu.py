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

from typing import Dict, List, Optional

from .utils import render_conversation_for_completion, render_mc

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


def format_mmlu_conversation(
    question: str, choices: List[str], answer: int, subject: str
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
        Example:
        {
            "messages": [
                {"role": "user", "content": "Multiple Choice question: ..."},
                {"role": "assistant", "content": "B"}
            ],
            "letters": ["A", "B", "C", "D"],
            "subject": "college_biology"
        }
    """
    # Validate inputs
    assert len(choices) == 4, f"MMLU should have 4 choices, got {len(choices)}"
    assert 0 <= answer < 4, f"MMLU answer index should be 0-3, got {answer}"

    # Render the question in multiple-choice format
    user_message = render_mc(question, MMLU_LETTERS, choices)

    # Get the answer letter
    answer_letter = MMLU_LETTERS[answer]

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer_letter},
    ]

    conversation = {
        "messages": messages,
        "letters": MMLU_LETTERS,  # Useful for evaluation
        "subject": subject,  # Useful for per-subject metrics
    }

    return conversation


def evaluate_mmlu(ground_truth_answer: str, predicted_text: str) -> bool:
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


def load_mmlu_from_hf(
    subset: str = "all",
    split: str = "test",
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = "/sensei-fs/users/divgoyal/nanochat_midtraining_data",
    shuffle_seed: int = 42,
) -> List[Dict]:
    """
    Load MMLU dataset from HuggingFace.

    Args:
        subset: Dataset subset to load:
                - "all" for all subjects combined
                - "auxiliary_train" for the auxiliary training set
                - Or a specific subject name (e.g., "college_biology")
        split: Dataset split to load:
               - "test" for the test set (most common)
               - "validation" for validation set
               - "dev" for development set (5 examples per subject)
               - "train" for training set (only with auxiliary_train subset)
        max_examples: Optional limit on number of examples to load
        cache_dir: Directory to cache the downloaded dataset
        shuffle_seed: Random seed for shuffling (default: 42 for reproducibility)

    Returns:
        List of examples, each with 'question', 'choices', 'answer', 'subject', 'conversation' keys

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

    # Validate inputs
    if subset != "all" and subset != "auxiliary_train":
        assert (
            subset in MMLU_SUBJECTS
        ), f"MMLU subset must be 'all', 'auxiliary_train', or one of {len(MMLU_SUBJECTS)} subjects, got: {subset}"

    if subset == "auxiliary_train":
        assert split == "train", "auxiliary_train subset only has 'train' split"
    else:
        assert split in [
            "train",
            "validation",
            "dev",
            "test",
        ], f"MMLU split must be 'train', 'validation', 'dev', or 'test', got: {split}"

    # Load from HuggingFace with specified cache directory
    dataset = load_dataset("cais/mmlu", subset, split=split, cache_dir=cache_dir)

    # Handle auxiliary_train's nested structure
    if subset == "auxiliary_train":
        # The auxiliary_train rows have a weird additional 'train' wrapper
        dataset = dataset.map(lambda row: row["train"], remove_columns=["train"])

    # Shuffle for variety (deterministic with seed)
    dataset = dataset.shuffle(seed=shuffle_seed)

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
                "conversation": format_mmlu_conversation(
                    question, choices, answer, subject
                ),
            }
        )

    return examples


def setup_mmlu_task(
    evaluator,
    tokenizer,
    subset: str = "all",
    split: str = "test",
    cache_dir: Optional[str] = "/sensei-fs/users/divgoyal/nanochat_midtraining_data",
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

    def load_fn(max_examples=None):
        """Load MMLU data."""
        return load_mmlu_from_hf(
            subset=subset,
            split=split,
            max_examples=max_examples,
            cache_dir=cache_dir,
        )

    def eval_fn(example, generated_text, return_details=False):
        """Evaluate an MMLU prediction."""
        ground_truth_letter = example["answer_letter"]
        is_correct = evaluate_mmlu(ground_truth_letter, generated_text)

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
