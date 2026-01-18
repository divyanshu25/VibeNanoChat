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

import re
from typing import Dict, List, Optional

# Regex pattern to extract the numerical answer after #### marker
GSM_ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the numerical answer from a GSM8K response.

    Looks for the #### marker followed by a number. Normalizes by removing commas.
    Follows the official GSM8K evaluation code:
    https://github.com/openai/grade-school-math/blob/master/grade_school_math/dataset.py

    Args:
        text: Response text that may contain #### marker with answer

    Returns:
        The extracted numerical answer as a string, or None if not found

    Examples:
        >>> extract_answer("The answer is #### 42")
        '42'
        >>> extract_answer("Total: $<<10*5=50>>50\\n#### 50")
        '50'
        >>> extract_answer("The result is #### 1,234.5")
        '1234.5'
    """
    match = GSM_ANSWER_RE.search(text)
    if match:
        answer_str = match.group(1).strip()
        # Remove commas from numbers (e.g., "1,234" -> "1234")
        answer_str = answer_str.replace(",", "")
        return answer_str
    return None


def parse_gsm8k_answer(answer_text: str) -> List[Dict[str, str]]:
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

    Example:
        >>> answer = "She earns 12/60 = $<<12/60=0.2>>0.2 per minute. #### 10"
        >>> parts = parse_gsm8k_answer(answer)
        >>> parts[0]
        {'type': 'text', 'text': 'She earns 12/60 = $'}
        >>> parts[1]
        {'type': 'python', 'text': '12/60'}
        >>> parts[2]
        {'type': 'output_start', 'text': '0.2'}
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


def format_gsm8k_conversation(question: str, answer: str) -> Dict:
    """
    Format a GSM8K example as a conversation.

    Args:
        question: The math problem question
        answer: The step-by-step answer with tool calls and final answer

    Returns:
        Conversation dict with 'messages' key containing user and assistant messages
        Example:
        {
            "messages": [
                {"role": "user", "content": "What is 2+2?"},
                {"role": "assistant", "content": [{"type": "text", "text": "The answer is "}, {"type": "python", "text": "2+2"}, {"type": "output_start", "text": "4"}]},
            ]
        }
    """
    # Parse the answer into structured parts
    answer_parts = parse_gsm8k_answer(answer)

    # Create conversation in chat format
    messages = [
        {"role": "user", "content": question},
        {"role": "assistant", "content": answer_parts},
    ]

    return {"messages": messages}


def evaluate_gsm8k(ground_truth_answer: str, predicted_text: str) -> bool:
    """
    Evaluate a GSM8K prediction against the ground truth.

    Extracts the numerical answer from both ground truth and prediction,
    then compares them for exact match.

    Args:
        ground_truth_answer: The full ground truth answer with #### marker
        predicted_text: The model's generated response text

    Returns:
        True if the predicted answer matches ground truth, False otherwise

    Examples:
        >>> evaluate_gsm8k("Working through: #### 42", "I think #### 42")
        True
        >>> evaluate_gsm8k("The answer is #### 100", "Result: #### 99")
        False
        >>> evaluate_gsm8k("#### 1,234", "#### 1234")
        True
    """
    # Extract numerical answers
    ref_answer = extract_answer(ground_truth_answer)
    pred_answer = extract_answer(predicted_text)

    # Both must have extractable answers to compare
    if ref_answer is None or pred_answer is None:
        return False

    # Compare as strings (already normalized by removing commas)
    return ref_answer == pred_answer


def load_gsm8k_from_hf(
    split: str = "test",
    max_examples: Optional[int] = None,
    cache_dir: Optional[str] = "/sensei-fs/users/divgoyal/nanochat_midtraining_data",
) -> List[Dict]:
    """
    Load GSM8K dataset from HuggingFace.

    Args:
        split: Dataset split to load ('train' or 'test')
        max_examples: Optional limit on number of examples to load
        cache_dir: Directory to cache the downloaded dataset (default: /sensei-fs/users/divgoyal/nanochat_midtraining_data)

    Returns:
        List of examples, each with 'question', 'answer', 'conversation' keys

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
    dataset = load_dataset("openai/gsm8k", "main", split=split, cache_dir=cache_dir)

    # Limit examples if requested
    if max_examples is not None:
        dataset = dataset.select(range(min(max_examples, len(dataset))))

    # Convert to our format
    examples = []
    for item in dataset:
        question = item["question"]
        answer = item["answer"]

        examples.append(
            {
                "question": question,
                "answer": answer,
                "conversation": format_gsm8k_conversation(question, answer),
            }
        )

    return examples


def load_gsm8k_from_jsonl(
    filepath: str, max_examples: Optional[int] = None
) -> List[Dict]:
    """
    Load GSM8K dataset from a JSONL file.

    Expected format: Each line is a JSON object with 'question' and 'answer' keys.

    Args:
        filepath: Path to JSONL file
        max_examples: Optional limit on number of examples to load

    Returns:
        List of examples, each with 'question', 'answer', 'conversation' keys
    """
    import json

    examples = []
    with open(filepath, "r") as f:
        for i, line in enumerate(f):
            if max_examples is not None and i >= max_examples:
                break

            item = json.loads(line)
            question = item["question"]
            answer = item["answer"]

            examples.append(
                {
                    "question": question,
                    "answer": answer,
                    "conversation": format_gsm8k_conversation(question, answer),
                }
            )

    return examples


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

    def load_fn(max_examples=None):
        """Load GSM8K data."""
        return load_gsm8k_from_hf(
            split=split, max_examples=max_examples, cache_dir=cache_dir
        )

    def eval_fn(example, generated_text):
        """Evaluate a GSM8K prediction."""
        ground_truth_answer = example["answer"]
        return evaluate_gsm8k(ground_truth_answer, generated_text)

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
