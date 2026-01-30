"""
Prepare TaskMixture datasets for mid-training.

This script pre-tokenizes SmolTalk, MMLU, and GSM8K datasets and saves them
as binary files for efficient loading during multi-GPU training.

Datasets:
- SmolTalk: General conversations (~460K train rows)
- MMLU auxiliary_train: Multiple choice problems from ARC, MC_TEST, OBQA, RACE (~100K train rows)
- GSM8K: Math reasoning (~8K train rows)

Total: ~568K training examples

Output:
- train.bin: Concatenated tokenized data for training
- val.bin: Concatenated tokenized data for validation
- metadata.json: Token statistics and special token mapping

Note:
    This script uses the reusable dataloaders from src/dataloaders/ (SmolTalkDataLoader,
    MMLUDataLoader, and GSM8KDataLoader) to ensure consistency with evaluation code and
    enable dataset reuse across different parts of the codebase.

Usage:
    python data/task_mixture/prepare.py

    # Or with custom output directory:
    python data/task_mixture/prepare.py --output_dir /path/to/output
"""

import argparse
import json
import os
import re
import shutil
import sys

import numpy as np
from datasets import concatenate_datasets
from tqdm import tqdm

# Add src to path so we can import from gpt_2 module
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))
from dataloaders.gsm8k_dataloader import GSM8KDataLoader
from dataloaders.mmlu_dataloader import MMLUDataLoader
from dataloaders.smoltalk_dataloader import SmolTalkDataLoader
from gpt_2.utils import get_custom_tokenizer

# Number of workers for parallel dataset processing (using HuggingFace datasets .map())
NUM_PROC = 8


# =============================================================================
# SECTION 1: DATASET FORMATTING FUNCTIONS
# =============================================================================
# Each dataset has its own structure. These functions convert raw examples
# into a unified chat format using our special tokens.
#
# Final format for all datasets:
#   <|bos|><|user_start|>...<|user_end|><|assistant_start|>...<|assistant_end|>


def format_smoltalk(example):
    """
    Format SmolTalk conversation to chat format.

    SmolTalk structure:
        {'messages': [{'role': 'user', 'content': '...'}, {'role': 'assistant', 'content': '...'}, ...]}

    Output format:
        <|bos|><|user_start|>...<|user_end|><|assistant_start|>...<|assistant_end|>...

    Note: Multi-turn conversations will have alternating user/assistant blocks.
    """
    messages = example["messages"]
    formatted_parts = ["<|bos|>"]  # Start with beginning-of-sequence token

    for msg in messages:
        role = msg["role"].lower()
        content = msg["content"]

        if role == "user":
            formatted_parts.append(f"<|user_start|>{content}<|user_end|>")
        elif role == "assistant":
            formatted_parts.append(f"<|assistant_start|>{content}<|assistant_end|>")
        else:
            # Handle other roles (system, etc.) as user for simplicity
            formatted_parts.append(f"<|user_start|>{content}<|user_end|>")

    return {"text": "".join(formatted_parts)}


def format_mmlu(example):
    """
    Format MMLU multiple-choice question to chat format.

    Uses the MMLUDataLoader formatting logic but adapted for training format
    (includes both question and answer, unlike evaluation format).

    Output format:
        <|bos|><|user_start|>Subject: physics

        What is the speed of light?

        A. 3x10^8 m/s
        B. 3x10^6 m/s
        C. 3x10^4 m/s
        D. 3x10^2 m/s<|user_end|><|assistant_start|>The answer is A. 3x10^8 m/s<|assistant_end|>
    """
    question = example["question"]
    choices = example["choices"]
    answer_idx = example["answer"]  # Integer index (0-3)
    subject = example.get("subject", "")

    # Format choices as A, B, C, D
    choice_labels = ["A", "B", "C", "D"]
    choices_text = "\n".join(
        [f"{label}. {choice}" for label, choice in zip(choice_labels, choices)]
    )

    # Get the correct answer text (e.g., "A. 3x10^8 m/s")
    correct_answer = f"{choice_labels[answer_idx]} : {choices[answer_idx]}"

    # Build the user prompt with optional subject
    if subject:
        user_content = f"Subject: {subject}\n\n{question}\n\n{choices_text}"
    else:
        user_content = f"{question}\n\n{choices_text}"

    # Combine into final chat format
    text = (
        f"<|bos|>"
        f"<|user_start|>{user_content}<|user_end|>"
        f"<|assistant_start|>{correct_answer}<|assistant_end|>"
    )
    return {"text": text}


def format_gsm8k(example):
    """
    Format GSM8K math problem to chat format with structured tool calls.

    Uses the GSM8KDataLoader parsing logic but adapted for training format
    (includes both question and full answer with tool calls).

    Output format:
        <|bos|><|user_start|>Janet has 3 apples...<|user_end|>
        <|assistant_start|>Janet has 3 apples. She buys 2 more
        <|python|>12/60<|python_end|><|output_start|>0.2<|output_end|>
        ... #### 5<|assistant_end|>

    Note: Tool calls are wrapped in special tokens so the model learns the structure.
    """
    question = example["question"]
    answer = example["answer"]  # Contains reasoning + "#### final_answer"

    # Use GSM8KDataLoader's parsing logic to parse tool calls
    # This ensures consistency with evaluation code
    formatted_answer_parts = []

    # Split on tool call markers: <<...>>
    segments = re.split(r"(<<[^>]+>>)", answer)

    for segment in segments:
        if segment.startswith("<<") and segment.endswith(">>"):
            # This is a calculator tool call
            inner = segment[2:-2]  # Remove << >>

            # Split on = to get expression and result
            if "=" in inner:
                expr, result = inner.rsplit("=", 1)
                # Format as structured tool call with special tokens
                formatted_answer_parts.append(
                    f"<|python|>{expr}<|python_end|>"
                    f"<|output_start|>{result}<|output_end|>"
                )
            else:
                # No = sign, just add the expression
                formatted_answer_parts.append(f"<|python|>{inner}<|python_end|>")
        else:
            # Regular text - add as is
            formatted_answer_parts.append(segment)

    # Combine into final answer text
    formatted_answer = "".join(formatted_answer_parts)

    text = (
        f"<|bos|>"
        f"<|user_start|>{question}<|user_end|>"
        f"<|assistant_start|>{formatted_answer}<|assistant_end|>"
    )
    return {"text": text}


# =============================================================================
# SECTION 4: DATASET LOADING AND COMBINATION
# =============================================================================
# Load each dataset from HuggingFace, apply formatting, and combine them.


def print_sample_example(dataset_name, example_text):
    """Print a sample example from a dataset for debugging/verification."""
    print(f"\n  {'─' * 50}")
    print(f"  📝 Sample from {dataset_name}:")
    print(f"  {'─' * 50}")
    # Truncate long examples for display
    display_text = (
        example_text[:600] + "..." if len(example_text) > 600 else example_text
    )
    # Indent each line for better formatting
    for line in display_text.split("\n"):
        print(f"  {line}")
    print(f"  {'─' * 50}")


def load_and_format_datasets(split, cache_dir=None):
    """
    Load all three datasets, format them, and combine into one.

    Args:
        split: "train" or "test"
        cache_dir: Directory to cache downloaded HuggingFace datasets

    Returns:
        Combined HuggingFace Dataset with all examples formatted
    """
    datasets = []

    # -------------------------------------------------------------------------
    # Dataset 1: SmolTalk - General conversations
    # -------------------------------------------------------------------------
    # SmolTalk has millions of examples, so we randomly sample 460K for training
    # to keep the dataset balanced with MMLU and GSM8K
    SMOLTALK_TRAIN_SAMPLES = 460_000
    SMOLTALK_TEST_SAMPLES = 5_000  # Smaller sample for validation

    print(f"\n📚 Loading SmolTalk ({split})...")
    try:
        # Use SmolTalkDataLoader for consistent loading
        target_samples = (
            SMOLTALK_TRAIN_SAMPLES if split == "train" else SMOLTALK_TEST_SAMPLES
        )

        smoltalk_loader = SmolTalkDataLoader(
            config="all", split=split, cache_dir=cache_dir, shuffle_seed=42
        )

        # Load data using the dataloader (returns list of dicts)
        # The dataloader handles shuffling and sampling
        smoltalk_examples = smoltalk_loader.load_data(max_examples=target_samples)

        print(
            f"  Loaded {len(smoltalk_examples):,} examples (sampled via SmolTalkDataLoader)"
        )

        # Convert back to HuggingFace dataset format for consistency
        from datasets import Dataset

        smoltalk = Dataset.from_list(smoltalk_examples)

        # Apply formatting function to convert to chat format
        # remove_columns removes original columns, keeping only 'text'
        smoltalk = smoltalk.map(
            format_smoltalk, remove_columns=smoltalk.column_names, num_proc=NUM_PROC
        )
        print(f"  ✓ SmolTalk: {len(smoltalk):,} examples")
        # Show one example for verification
        print_sample_example("SmolTalk", smoltalk[0]["text"])
        datasets.append(smoltalk)
    except Exception as e:
        print(f"  ✗ SmolTalk failed: {e}")

    # -------------------------------------------------------------------------
    # Dataset 2: MMLU - Multiple choice problems
    # -------------------------------------------------------------------------
    # MMLU auxiliary_train contains ~100K examples from various sources:
    # ARC (science), MC_TEST (reading), OBQA (science), RACE (reading)

    print(f"\n📚 Loading MMLU auxiliary_train ({split})...")
    try:
        # MMLU has subset="auxiliary_train" with split="train" for training (99,842 examples)
        # For validation, we use subset="all" with split="validation" (1,531 examples)
        if split == "train":
            mmlu_subset = "auxiliary_train"
            mmlu_split = "train"
        else:
            mmlu_subset = "all"
            mmlu_split = "validation"

        # Use MMLUDataLoader for consistent loading
        mmlu_loader = MMLUDataLoader(
            subset=mmlu_subset, split=mmlu_split, cache_dir=cache_dir, shuffle_seed=42
        )

        # Load data using the dataloader (returns list of dicts)
        mmlu_examples = mmlu_loader.load_data()

        # Convert back to HuggingFace dataset format for consistency
        from datasets import Dataset

        mmlu = Dataset.from_list(mmlu_examples)

        # Apply formatting to convert to chat format
        mmlu = mmlu.map(
            format_mmlu, remove_columns=mmlu.column_names, num_proc=NUM_PROC
        )
        print(
            f"  ✓ MMLU ({mmlu_split}): {len(mmlu):,} examples (loaded via MMLUDataLoader)"
        )

        print_sample_example("MMLU", mmlu[0]["text"])
        datasets.append(mmlu)
    except Exception as e:
        print(f"  ✗ MMLU failed: {e}")

    # -------------------------------------------------------------------------
    # Dataset 3: GSM8K - Math word problems with step-by-step solutions
    # -------------------------------------------------------------------------
    # GSM8K has ~8K training examples of grade school math problems

    print(f"\n📚 Loading GSM8K ({split})...")
    try:
        # Use GSM8KDataLoader for consistent loading
        gsm8k_loader = GSM8KDataLoader(
            split=split, cache_dir=cache_dir, shuffle_seed=42
        )

        # Load data using the dataloader (returns list of dicts)
        gsm8k_examples = gsm8k_loader.load_data()

        # Convert back to HuggingFace dataset format for consistency
        from datasets import Dataset

        gsm8k = Dataset.from_list(gsm8k_examples)

        # Apply formatting to convert to chat format
        gsm8k = gsm8k.map(
            format_gsm8k, remove_columns=gsm8k.column_names, num_proc=NUM_PROC
        )
        print(f"  ✓ GSM8K: {len(gsm8k):,} examples (loaded via GSM8KDataLoader)")

        print_sample_example("GSM8K", gsm8k[0]["text"])
        datasets.append(gsm8k)
    except Exception as e:
        print(f"  ✗ GSM8K failed: {e}")

    # -------------------------------------------------------------------------
    # Combine all datasets
    # -------------------------------------------------------------------------
    if not datasets:
        raise ValueError("No datasets were loaded successfully!")

    # concatenate_datasets joins all datasets into one
    combined = concatenate_datasets(datasets)

    # Shuffle like a deck of cards - interleave examples from all datasets
    print("\n🔀 Shuffling combined dataset...")
    combined = combined.shuffle(seed=42)
    print(f"\n✅ Combined & shuffled: {len(combined):,} total examples")

    return combined


# =============================================================================
# SECTION 5: TOKENIZATION AND SAVING TO BINARY
# =============================================================================
# Convert formatted text to token IDs and save as binary files for fast loading.


def tokenize_and_save(dataset, output_file, enc, special_tokens):
    """
    Tokenize all examples and save as a binary file.

    Args:
        dataset: HuggingFace Dataset with 'text' column
        output_file: Path to save binary file (e.g., "train.bin")
        enc: Custom tiktoken encoder with special tokens registered
        special_tokens: dict of special token mappings (for display purposes)

    Returns:
        int: Total number of tokens in the saved file
    """
    print(f"\n🔄 Tokenizing and saving to {output_file}...")

    # -------------------------------------------------------------------------
    # Debug: Print sample examples before and after tokenization
    # -------------------------------------------------------------------------
    print("\n" + "=" * 60)
    print("📋 SAMPLE EXAMPLES (before and after tokenization)")
    print("=" * 60)

    num_samples = min(3, len(dataset))
    for i in range(num_samples):
        example = dataset[i]
        text = example["text"]
        # Use tiktoken's native special token support!
        # allowed_special="all" tells tiktoken to recognize all registered special tokens
        tokens = enc.encode(text, allowed_special="all")

        print(f"\n{'─' * 60}")
        print(f"🔹 Example {i + 1}")
        print(f"{'─' * 60}")

        # Show original text (truncated if too long)
        print(f"\n📝 BEFORE (raw text, {len(text)} chars):")
        display_text = text[:500] + "..." if len(text) > 500 else text
        print(display_text)

        # Show token IDs (first 100)
        print(f"\n🔢 AFTER (token IDs, {len(tokens)} tokens):")
        display_tokens = tokens[:100]
        print(display_tokens)
        if len(tokens) > 100:
            print(f"   ... ({len(tokens) - 100} more tokens)")

        # Decode tokens back to text to verify encoding is correct
        # The custom encoder can decode special tokens natively!
        print("\n🔄 DECODED (tokens → text):")
        decoded_text = enc.decode(display_tokens[:50])
        # Show special tokens in brackets for visibility
        for token_str in special_tokens.keys():
            decoded_text = decoded_text.replace(token_str, f"[{token_str}]")
        print(decoded_text)
        if len(display_tokens) > 50:
            print("   ... (truncated)")

    print("\n" + "=" * 60)
    print("End of samples")
    print("=" * 60 + "\n")

    # -------------------------------------------------------------------------
    # Tokenize all examples using dataset.map()
    # -------------------------------------------------------------------------
    def tokenize_example(example):
        tokens = enc.encode(example["text"], allowed_special="all")
        return {"ids": tokens, "len": len(tokens)}

    print("  Tokenizing dataset...")
    tokenized = dataset.map(
        tokenize_example,
        remove_columns=["text"],
        desc="Tokenizing",
        num_proc=NUM_PROC,
    )

    # Calculate total tokens
    total_tokens = sum(tokenized["len"])
    print(f"  Total tokens: {total_tokens:,}")

    # -------------------------------------------------------------------------
    # Save to binary file using memmap (memory efficient)
    # -------------------------------------------------------------------------
    # Use uint16 (max 65535) - sufficient for GPT-2 vocab (~50k) + custom tokens
    dtype = np.uint16

    # Create memory-mapped file - writes directly to disk, not RAM
    arr = np.memmap(output_file, dtype=dtype, mode="w+", shape=(total_tokens,))

    # Write in batches to avoid memory overflow
    total_batches = min(1024, len(tokenized))  # Don't use more batches than examples
    idx = 0
    for batch_idx in tqdm(range(total_batches), desc="Writing to disk"):
        # Get one batch (1/total_batches of dataset)
        batch = tokenized.shard(
            num_shards=total_batches, index=batch_idx, contiguous=True
        ).with_format("numpy")

        # Concatenate all token IDs in this batch
        arr_batch = np.concatenate(batch["ids"])

        # Write batch to file and advance position
        arr[idx : idx + len(arr_batch)] = arr_batch
        idx += len(arr_batch)

    # Ensure all data is written to disk
    arr.flush()

    size_mb = os.path.getsize(output_file) / (1024 * 1024)
    print(f"  Saved {output_file} ({size_mb:.2f} MB)")

    return total_tokens


# =============================================================================
# SECTION 6: MAIN ENTRY POINT
# =============================================================================


def main():
    """Main function to prepare the TaskMixture dataset."""

    # -------------------------------------------------------------------------
    # Parse command line arguments
    # -------------------------------------------------------------------------
    parser = argparse.ArgumentParser(description="Prepare TaskMixture datasets")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/mnt/localssd/VibeNanoChat/data/task_mixture",  # Local disk (mmap compatible)
        help="Local directory to save processed data (must support mmap)",
    )
    parser.add_argument(
        "--final_dir",
        type=str,
        default="/sensei-fs/users/divgoyal/nanochat_midtraining_data",
        help="Final destination to copy files to (can be network filesystem)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/sensei-fs/users/divgoyal/nanochat_midtraining_data/hf_cache",
        help="Directory to cache downloaded HuggingFace datasets",
    )
    args = parser.parse_args()

    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)

    print("=" * 80)
    print("🔧 TaskMixture Dataset Preparation")
    print("=" * 80)

    # -------------------------------------------------------------------------
    # Initialize custom tokenizer with special tokens
    # -------------------------------------------------------------------------
    # Create a custom tiktoken encoder that has our special tokens registered
    # This allows us to use enc.encode(text, allowed_special="all") directly
    enc, special_tokens = get_custom_tokenizer()

    print("\n📝 Tokenizer info:")
    print(
        f"   Vocab size (with special tokens): {enc.n_vocab}"
    )  # 50262 (50257 base + 5 new)
    print(f"   Custom special tokens: {len(special_tokens)}")  # 5 new tokens
    print(f"   Special tokens: {list(special_tokens.keys())}")

    # -------------------------------------------------------------------------
    # Prepare metadata to save
    # -------------------------------------------------------------------------
    # Note: enc.n_vocab now includes both base GPT-2 tokens and our special tokens
    base_vocab_size = 50257  # Original GPT-2 vocab size
    metadata = {
        "base_vocab_size": base_vocab_size,
        "extended_vocab_size": enc.n_vocab,  # Already includes special tokens
        "special_tokens": special_tokens,
        "dtype": "uint16",  # Important for loading the binary files correctly
        "splits": {},
    }

    # -------------------------------------------------------------------------
    # Process train and test splits
    # -------------------------------------------------------------------------
    for split in ["train", "test"]:
        print(f"\n{'=' * 80}")
        print(f"📊 Processing {split} split...")
        print("=" * 80)

        try:
            # Step 1: Load and format all datasets for this split
            dataset = load_and_format_datasets(split, args.cache_dir)

            # Step 2: Tokenize and save to binary file
            # Note: "test" split is saved as "val.bin" (common naming convention)
            split_name = "val" if split == "test" else split
            output_file = os.path.join(args.output_dir, f"{split_name}.bin")
            num_tokens = tokenize_and_save(dataset, output_file, enc, special_tokens)

            # Step 3: Record statistics in metadata
            metadata["splits"][split_name] = {
                "num_examples": len(dataset),
                "num_tokens": num_tokens,
                "file": f"{split_name}.bin",
            }
        except Exception as e:
            print(f"  ✗ Failed to process {split}: {e}")
            import traceback

            traceback.print_exc()

    # -------------------------------------------------------------------------
    # Save metadata JSON
    # -------------------------------------------------------------------------
    metadata_file = os.path.join(args.output_dir, "metadata.json")
    with open(metadata_file, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n📋 Saved metadata to {metadata_file}")

    # -------------------------------------------------------------------------
    # Print summary
    # -------------------------------------------------------------------------
    print("\n" + "=" * 80)
    print("✅ PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\nOutput directory: {args.output_dir}")
    print("\nFiles created:")
    for split_name, info in metadata["splits"].items():
        print(
            f"  - {info['file']}: {info['num_tokens']:,} tokens from {info['num_examples']:,} examples"
        )
    print("  - metadata.json")
    print(
        "\nTo use in training, update TaskMixtureDataloader to read from these .bin files."
    )

    # -------------------------------------------------------------------------
    # Copy files to final destination (network filesystem) and delete local copy
    # -------------------------------------------------------------------------
    if args.final_dir and args.final_dir != args.output_dir:
        print(f"\n📦 Copying files to {args.final_dir}...")
        os.makedirs(args.final_dir, exist_ok=True)

        files_to_copy = ["train.bin", "val.bin", "metadata.json"]
        for filename in files_to_copy:
            src = os.path.join(args.output_dir, filename)
            dst = os.path.join(args.final_dir, filename)
            if os.path.exists(src):
                print(f"  Copying {filename}...")
                shutil.copy2(src, dst)
                print(f"  ✓ {filename} copied")
                # Delete local copy
                os.remove(src)
                print(f"  🗑️  {filename} deleted from local")
            else:
                print(f"  ⚠️  {filename} not found, skipping")

        print(f"\n✅ Files copied to {args.final_dir} and local copies deleted")


# =============================================================================
# Run main() when script is executed directly
# =============================================================================
if __name__ == "__main__":
    main()
