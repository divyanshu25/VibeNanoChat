"""
Word utilities for dataloaders.

This module provides shared utilities for working with word lists across
different dataloaders.
"""

import os
import urllib.request
from typing import List, Optional

# A list of 370K English words
WORD_LIST_URL = "https://raw.githubusercontent.com/dwyl/english-words/refs/heads/master/words_alpha.txt"


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
