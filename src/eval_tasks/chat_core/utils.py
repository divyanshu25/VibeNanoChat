"""
Utility functions for ChatCORE evaluation tasks.

Provides helper functions to:
- Render prompts from conversations
- Setup evaluators with common configurations
- Register tasks with the evaluator

Performance Note:
==================
The ChatCORE evaluator uses KV (Key-Value) caching for efficient generation:
- Without KV cache: Each token requires reprocessing all previous tokens → O(N²)
- With KV cache: Each token reuses cached attention keys/values → O(N)
- Result: 5-10x faster generation with ~2x memory usage

This is implemented transparently in the evaluator's generate methods.
"""

from typing import Dict, List, Optional


def render_conversation_for_completion(conversation: Dict, tokenizer) -> List[int]:
    """
    Render a conversation into tokens for completion using special token format.

    Takes a conversation dict with messages and renders it into a prompt
    that ends right before where the assistant should respond, using the
    special token format that matches training:
    <|bos|><|user_start|>...<|user_end|><|assistant_start|>

    Args:
        conversation: Dict with 'messages' key containing list of messages.
                     Each message has 'role' and 'content' keys.
        tokenizer: Tokenizer with encode method (supports allowed_special="all")

    Returns:
        List of token IDs representing the prompt

    Example:
        >>> conversation = {
        ...     'messages': [
        ...         {'role': 'user', 'content': 'What is 2+2?'},
        ...         {'role': 'assistant', 'content': ...}  # Will be predicted
        ...     ]
        ... }
        >>> tokens = render_conversation_for_completion(conversation, tokenizer)
        # Returns tokens for: "<|bos|><|user_start|>What is 2+2?<|user_end|><|assistant_start|>"
    """
    messages = conversation["messages"]
    formatted_parts = ["<|bos|>"]  # Start with beginning-of-sequence token

    for i, msg in enumerate(messages):
        role = msg["role"].lower()
        content = msg["content"]

        # Extract text content from structured or string format
        if isinstance(content, str):
            text_content = content
        else:
            # If content is structured (list of parts), extract text
            text_parts = [p["text"] for p in content if p.get("type") == "text"]
            text_content = "".join(text_parts)

        # For assistant messages, add the start token but not the content
        # (we only want to render up to where assistant should respond)
        if role == "assistant":
            formatted_parts.append("<|assistant_start|>")
            break
        elif role == "user":
            # Add user message with special tokens
            formatted_parts.append(f"<|user_start|>{text_content}<|user_end|>")
        else:
            # Handle other roles (system, etc.) as user for consistency with training
            formatted_parts.append(f"<|user_start|>{text_content}<|user_end|>")

    # Combine into final prompt text
    prompt_text = "".join(formatted_parts)

    # Encode to tokens with special token support
    # The allowed_special="all" parameter tells the tokenizer to recognize
    # our custom special tokens (<|bos|>, <|user_start|>, etc.)
    tokens = tokenizer.encode(prompt_text, allowed_special="all")
    return tokens


def render_mc(question: str, letters: List[str], choices: List[str]) -> str:
    """
    Render a multiple-choice question in a standardized format.

    This follows the same format used in nanochat. Key design decisions:
    1. Letter appears AFTER the choice text for better token binding in small models
    2. No whitespace between delimiter (=) and letter to ensure correct tokenization

    Args:
        question: The question text
        letters: List of answer letters (e.g., ["A", "B", "C", "D"])
        choices: List of choice texts corresponding to each letter

    Returns:
        Formatted multiple-choice question string

    Example:
        >>> render_mc("What color is the sky?", ["A", "B"], ["Blue", "Red"])
        'Multiple Choice question: What color is the sky?\\n- Blue=A\\n- Red=B\\n\\nRespond only with the letter of the correct answer.'
    """
    query = f"Multiple Choice question: {question}\n"
    query += "".join(
        [f"- {choice}={letter}\n" for letter, choice in zip(letters, choices)]
    )
    query += "\nRespond only with the letter of the correct answer."
    return query


def format_arc_conversation(
    question: str, choices_text: List[str], choices_labels: List[str], answer_key: str
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
        Example:
        {
            "messages": [
                {"role": "user", "content": "Multiple Choice question: ..."},
                {"role": "assistant", "content": "A"}
            ],
            "letters": ["A", "B", "C", "D"]
        }
    """
    # Render the question in multiple-choice format
    user_message = render_mc(question, choices_labels, choices_text)

    messages = [
        {"role": "user", "content": user_message},
        {"role": "assistant", "content": answer_key},
    ]

    conversation = {
        "messages": messages,
        "letters": choices_labels,  # Useful for evaluation and letter constraint
    }
    return conversation


def evaluate_arc(ground_truth_letter: str, predicted_text: str) -> bool:
    """
    Evaluate an ARC prediction against the ground truth.

    Extracts the first letter from the predicted text and compares it to
    the ground truth answer letter. This is more lenient than requiring
    an exact match, as it handles cases where the model may generate
    additional text.

    Args:
        ground_truth_letter: The correct answer letter (e.g., "A")
        predicted_text: The model's generated response text

    Returns:
        True if the predicted answer matches ground truth, False otherwise

    Examples:
        >>> evaluate_arc("A", "A")
        True
        >>> evaluate_arc("B", "B is the correct answer")
        True
        >>> evaluate_arc("C", "The answer is D")
        False
        >>> evaluate_arc("A", "")
        False
    """
    if not predicted_text:
        return False

    # Extract the first character (should be the letter)
    # Strip whitespace and take the first character
    predicted_text = predicted_text.strip()
    if not predicted_text:
        return False

    predicted_letter = predicted_text[0].upper()

    return predicted_letter == ground_truth_letter.upper()


def load_arc_from_hf(
    subset: str = "ARC-Easy",
    split: str = "test",
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = "/sensei-fs/users/divgoyal/nanochat_midtraining_data",
    shuffle_seed: int = 42,
) -> List[Dict]:
    """
    Load ARC dataset from HuggingFace.

    Args:
        subset: Dataset subset ('ARC-Easy' or 'ARC-Challenge')
        split: Dataset split to load ('train', 'validation', or 'test')
        max_examples: Optional limit on number of examples to load
        cache_dir: Directory to cache the downloaded dataset
        shuffle_seed: Random seed for shuffling (default: 42 for reproducibility)

    Returns:
        List of examples, each with 'question', 'answer', 'choices', 'conversation' keys

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

    # Load from HuggingFace with specified cache directory
    dataset = load_dataset("allenai/ai2_arc", subset, split=split, cache_dir=cache_dir)

    # Shuffle for variety (deterministic with seed)
    dataset = dataset.shuffle(seed=shuffle_seed)

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
                "conversation": format_arc_conversation(
                    question, choices_text, choices_labels, answer_key
                ),
            }
        )

    return examples
