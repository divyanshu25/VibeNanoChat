#!/usr/bin/env python3
"""
Check token IDs in the data files to ensure they're within vocabulary bounds.
"""

import sys

import numpy as np


def check_token_ids(data_file, vocab_size=50266):
    """Check if any token IDs exceed the vocabulary size."""
    print(f"\n{'='*80}")
    print(f"Checking token IDs in: {data_file}")
    print(f"Expected vocab size: {vocab_size}")
    print(f"{'='*80}\n")

    try:
        # Load the binary file (with fallback for network filesystems)
        try:
            tokens = np.memmap(data_file, dtype=np.uint16, mode="r")
        except OSError as e:
            print(f"Warning: mmap failed ({e}), loading entire file into memory...")
            with open(data_file, "rb") as f:
                tokens = np.fromfile(f, dtype=np.uint16)

        print(f"Total tokens: {len(tokens):,}")

        # Get statistics
        min_token = tokens.min()
        max_token = tokens.max()

        print(f"Min token ID: {min_token}")
        print(f"Max token ID: {max_token}")
        print()

        # Check if max token exceeds vocab size
        if max_token >= vocab_size:
            print("❌ ERROR: Found token IDs that exceed vocab_size!")
            print(f"   Max token ID: {max_token}")
            print(f"   Vocab size: {vocab_size}")
            print("   Tokens will cause index out of bounds!")
            print()

            # Find how many tokens are out of bounds
            invalid_mask = tokens >= vocab_size
            num_invalid = invalid_mask.sum()
            print(
                f"   Invalid tokens: {num_invalid:,} ({num_invalid/len(tokens)*100:.4f}%)"
            )

            # Show some examples of invalid tokens
            invalid_indices = np.where(invalid_mask)[0][:10]
            print("\n   First 10 invalid token positions:")
            for idx in invalid_indices:
                print(f"      Position {idx}: token_id={tokens[idx]}")

        else:
            print(f"✅ All token IDs are within bounds (< {vocab_size})")

        # Show distribution of top token IDs
        print("\nToken ID distribution (top 20 highest):")
        unique, counts = np.unique(tokens, return_counts=True)
        # Get the 20 highest token IDs
        top_indices = np.argsort(unique)[-20:]
        for idx in top_indices[::-1]:
            token_id = unique[idx]
            count = counts[idx]
            status = "❌ OUT OF BOUNDS" if token_id >= vocab_size else "✓"
            print(f"   Token {token_id}: {count:,} occurrences {status}")

        print(f"\n{'='*80}\n")

    except Exception as e:
        print(f"❌ Error reading file: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python check_token_ids.py <data_file> [vocab_size]")
        print("Example: python check_token_ids.py /path/to/train.bin 50266")
        sys.exit(1)

    data_file = sys.argv[1]
    vocab_size = int(sys.argv[2]) if len(sys.argv) > 2 else 50266

    check_token_ids(data_file, vocab_size)


# uv run python check_token_ids.py  /sensei-fs/users/divgoyal/nanochat_midtraining_data/train.bin 50266
