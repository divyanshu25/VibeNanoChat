"""
Prepare HellaSwag dataset for language model evaluation.

HellaSwag is a commonsense reasoning benchmark where the model must choose
the most plausible continuation from 4 options given a context.

Dataset format:
- context: Activity/situation description
- endings: List of 4 possible continuations
- label: Index of correct ending (0-3)

Output: Processed JSON files ready for evaluation
"""

import os
import json
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Initialize GPT-2 tokenizer for preprocessing
enc = tiktoken.get_encoding("gpt2")


def process_example(example):
    """
    Process a single HellaSwag example.

    Returns a dictionary with:
    - context: The activity/situation text
    - endings: List of 4 possible continuations
    - label: Correct answer index (0-3)
    - context_tokens: Tokenized context
    - ending_tokens: List of tokenized endings
    """
    # Get context - combine activity label and context
    ctx = example["ctx"]

    # Some examples have activity_label, include it for context
    if "activity_label" in example and example["activity_label"]:
        ctx = example["activity_label"] + ": " + ctx

    # Get endings
    endings = example["endings"]

    # Get correct label
    label = int(example["label"]) if example["label"] != "" else -1

    # Tokenize for quick evaluation later
    ctx_tokens = enc.encode(ctx)
    ending_tokens = [enc.encode(" " + ending) for ending in endings]

    return {
        "context": ctx,
        "endings": endings,
        "label": label,
        "context_tokens": ctx_tokens,
        "ending_tokens": ending_tokens,
    }


def calculate_statistics(processed_data):
    """Calculate and print dataset statistics."""
    num_examples = len(processed_data)

    ctx_lengths = [len(ex["context_tokens"]) for ex in processed_data]
    ending_lengths = [
        len(ending) for ex in processed_data for ending in ex["ending_tokens"]
    ]

    print(f"\nDataset Statistics:")
    print(f"  Number of examples: {num_examples}")
    print(
        f"  Context length - mean: {np.mean(ctx_lengths):.1f}, "
        f"median: {np.median(ctx_lengths):.1f}, "
        f"max: {np.max(ctx_lengths)}"
    )
    print(
        f"  Ending length - mean: {np.mean(ending_lengths):.1f}, "
        f"median: {np.median(ending_lengths):.1f}, "
        f"max: {np.max(ending_lengths)}"
    )

    # Label distribution (should be ~25% each)
    labels = [ex["label"] for ex in processed_data if ex["label"] >= 0]
    unique, counts = np.unique(labels, return_counts=True)
    print(f"  Label distribution: {dict(zip(unique, counts))}")


if __name__ == "__main__":
    print("Downloading HellaSwag dataset...")

    # Load HellaSwag from HuggingFace
    # Dataset has ~70K examples total
    dataset = load_dataset("Rowan/hellaswag")

    # Process each split
    output_dir = "/sensei-fs/users/divgoyal/hellaswag"
    os.makedirs(output_dir, exist_ok=True)

    for split_name in dataset.keys():
        print(f"\nProcessing {split_name} split...")
        split = dataset[split_name]

        processed = []
        for example in tqdm(split, desc=f"Processing {split_name}"):
            processed.append(process_example(example))

        # Save as JSON for easy loading during evaluation
        output_file = os.path.join(output_dir, f"{split_name}.json")
        with open(output_file, "w") as f:
            json.dump(processed, f)

        print(f"Saved {len(processed)} examples to {output_file}")
        calculate_statistics(processed)

    print("\n✓ HellaSwag dataset preparation complete!")
    print(f"\nOutput directory: {output_dir}")
    print("Files created:")
    for split_name in dataset.keys():
        print(f"  - {split_name}.json")
