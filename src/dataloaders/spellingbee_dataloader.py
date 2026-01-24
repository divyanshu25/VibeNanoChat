"""
SpellingBee DataLoader

This module provides a data loader for the SpellingBee task: counting occurrences
of a letter in a word. The task helps models learn character-level understanding
and systematic counting.

Example:
    >>> loader = SpellingBeeDataLoader(size=100, split="train")
    >>> examples = loader.load_data()
    >>> print(f"Loaded {len(examples)} SpellingBee examples")
"""

import os
import random
import re
import urllib.request
from typing import Dict, List, Optional

# Letters of the alphabet
LETTERS = "abcdefghijklmnopqrstuvwxyz"

# A list of 370K English words
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"

# Separate train and test with different random seeds
TEST_RANDOM_SEED_OFFSET = 10_000_000

# Answer extraction pattern (same as GSM8K)
ANSWER_RE = re.compile(r"#### (\-?[0-9\.\,]+)")

# User message templates for data augmentation
USER_MSG_TEMPLATES = [
    "How many {letter} are in the word {word}",
    "How many {letter} are in {word}",
    "Count the number of {letter} in {word}",
    "How many times does {letter} appear in {word}",
    "What's the count of {letter} in {word}",
    "In the word {word}, how many {letter} are there",
    "How many letter {letter} are in the word {word}",
    "Count how many {letter} appear in {word}",
    "Tell me the number of {letter} in {word}",
    "How many occurrences of {letter} are in {word}",
    "Find the count of {letter} in {word}",
    "Can you count the {letter} letters in {word}",
    "What is the frequency of {letter} in {word}",
    "How many {letter}s are in {word}",
    "How many {letter}'s are in {word}",
    "Count all the {letter} in {word}",
    "How many times is {letter} in {word}",
    "Number of {letter} in {word}",
    "Total count of {letter} in {word}",
    "How many {letter} does {word} have",
    "How many {letter} does {word} contain",
    "What's the number of {letter} in {word}",
    "{word} has how many {letter}",
    "In {word}, count the {letter}",
    "How many {letter} appear in {word}",
    "Count the {letter} in {word}",
    "Give me the count of {letter} in {word}",
    "How many instances of {letter} in {word}",
    "Show me how many {letter} are in {word}",
    "Calculate the number of {letter} in {word}",
    # Spanish
    "¿Cuántas {letter} hay en {word}?",
    "¿Cuántas veces aparece {letter} en {word}?",
    "Cuenta las {letter} en {word}",
    "¿Cuántas letras {letter} tiene {word}?",
    # Chinese (Simplified)
    "{word}中有多少个{letter}",
    "{word}里有几个{letter}",
    "数一下{word}中的{letter}",
    "{word}这个词里有多少{letter}",
    # Korean
    "{word}에 {letter}가 몇 개 있나요",
    "{word}에서 {letter}의 개수는",
    "{word}에 {letter}가 몇 번 나오나요",
    "{word}라는 단어에 {letter}가 몇 개",
    # French
    "Combien de {letter} dans {word}",
    "Combien de fois {letter} apparaît dans {word}",
    "Compte les {letter} dans {word}",
    # German
    "Wie viele {letter} sind in {word}",
    "Wie oft kommt {letter} in {word} vor",
    "Zähle die {letter} in {word}",
    # Japanese
    "{word}に{letter}は何個ありますか",
    "{word}の中に{letter}がいくつ",
    "{word}に{letter}が何回出てくる",
]


def download_word_list(cache_dir: Optional[str] = None) -> List[str]:
    """
    Download and cache the English word list.

    Args:
        cache_dir: Optional directory to cache the word list

    Returns:
        List of English words
    """
    # Determine cache location - use repo's data/cache directory
    if cache_dir is None:
        # Get the repo root (assuming this file is in src/dataloaders/)
        repo_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        cache_dir = os.path.join(repo_root, "data", "cache")

    os.makedirs(cache_dir, exist_ok=True)

    filename = WORD_LIST_URL.split("/")[-1]
    cache_path = os.path.join(cache_dir, filename)

    # Download if not cached
    if not os.path.exists(cache_path):
        print(f"Downloading word list from {WORD_LIST_URL}...")
        urllib.request.urlretrieve(WORD_LIST_URL, cache_path)
        print(f"Saved to {cache_path}")

    # Load words
    with open(cache_path, "r", encoding="utf-8") as f:
        words = [line.strip() for line in f if line.strip()]

    return words


def extract_answer(text: str) -> Optional[str]:
    """
    Extract the numerical answer after #### marker.

    Args:
        text: Response text that may contain #### marker with answer

    Returns:
        The extracted numerical answer as a string, or None if not found
    """
    match = ANSWER_RE.search(text)
    if match:
        match_str = match.group(1).strip()
        match_str = match_str.replace(",", "")
        return match_str
    return None


class SpellingBeeDataLoader:
    """
    A data loader for the SpellingBee task (counting letters in words).

    This loader generates synthetic examples on-the-fly where the model
    must count how many times a letter appears in a word. The examples
    include both manual counting and Python verification.

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
        Initialize the SpellingBee data loader.

        Args:
            size: Number of examples to generate
            split: Dataset split to use ('train' or 'test')
            cache_dir: Directory to cache the word list
        """
        assert split in ["train", "test"], "SpellingBee split must be 'train' or 'test'"

        self.size = size
        self.split = split
        self.cache_dir = cache_dir

        # Load word list
        self.words = download_word_list(cache_dir)

    def load_data(self, max_examples: Optional[int] = None) -> List[Dict]:
        """
        Generate SpellingBee examples.

        Args:
            max_examples: Optional limit on number of examples (if None, uses self.size)

        Returns:
            List of examples, each with 'word', 'letter', 'count', 'messages' keys
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
        Generate a single SpellingBee example.

        Args:
            index: Example index (used as random seed)

        Returns:
            Dictionary with example data
        """
        seed = index if self.split == "train" else TEST_RANDOM_SEED_OFFSET + index
        rng = random.Random(seed)

        # Pick a random word
        word = rng.choice(self.words)

        # Pick a letter from it (90%) or a random letter (10%)
        letter = rng.choice(word) if rng.random() < 0.9 else rng.choice(LETTERS)

        # Get the correct answer by simply counting
        count = word.count(letter)

        # Create a user message, with variations as data augmentation
        template = rng.choice(USER_MSG_TEMPLATES)

        # 30% chance to lowercase the template
        if rng.random() < 0.3:
            template = template.lower()

        quote_options = ["", "'", '"']
        letter_quote = rng.choice(quote_options)
        word_quote = rng.choice(quote_options)
        letter_wrapped = f"{letter_quote}{letter}{letter_quote}"
        word_wrapped = f"{word_quote}{word}{word_quote}"
        user_msg = template.format(letter=letter_wrapped, word=word_wrapped)

        if rng.random() < 0.5:  # 50% don't use question marks
            user_msg += "?"

        # Create the ideal assistant response
        assistant_parts = []
        word_letters = ",".join(list(word))

        manual_text = f"""We are asked to find the number '{letter}' in the word '{word}'. Let me try a manual approach first.

First spell the word out:
{word}:{word_letters}

Then count the occurrences of '{letter}':
"""

        # Simulate counting process
        running_count = 0
        for i, char in enumerate(word, 1):
            if char == letter:
                running_count += 1
                manual_text += f"{i}:{char} hit! count={running_count}\n"
            else:
                manual_text += f"{i}:{char}\n"

        manual_text += f"\nThis gives us {running_count}."
        assistant_parts.append({"type": "text", "text": manual_text})

        # Python verification
        assistant_parts.append(
            {"type": "text", "text": "\n\nLet me double check this using Python:\n\n"}
        )
        python_expr = f"'{word}'.count('{letter}')"
        assistant_parts.append({"type": "python", "text": python_expr})
        assistant_parts.append({"type": "output_start", "text": str(count)})

        # Final answer
        assistant_parts.append(
            {
                "type": "text",
                "text": f"\n\nPython gives us {count}.\n\nMy final answer is:\n\n#### {count}",
            }
        )

        # Create conversation
        messages = [
            {"role": "user", "content": user_msg},
            {"role": "assistant", "content": assistant_parts},
        ]

        conversation = {
            "word": word,
            "letter": letter,
            "count": count,
            "messages": messages,
        }

        return conversation

    @staticmethod
    def evaluate(ground_truth_answer: str, predicted_text: str) -> bool:
        """
        Evaluate a SpellingBee prediction against the ground truth.

        Args:
            ground_truth_answer: The correct answer with #### marker
            predicted_text: The model's generated response text

        Returns:
            True if the predicted answer matches ground truth, False otherwise
        """
        ref_num = extract_answer(ground_truth_answer)
        pred_num = extract_answer(predicted_text)

        if ref_num is None or pred_num is None:
            return False

        return ref_num == pred_num


if __name__ == "__main__":
    import json

    # Generate examples
    loader = SpellingBeeDataLoader(size=3, split="train")
    examples = loader.load_data()

    # Pretty-print the JSON
    for i, example in enumerate(examples, 1):
        print(f"\n{'='*80}")
        print(f"Example {i}")
        print(f"{'='*80}\n")
        print(json.dumps(example, indent=2, ensure_ascii=False))
        print()
