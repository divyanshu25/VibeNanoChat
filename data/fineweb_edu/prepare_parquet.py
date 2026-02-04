"""
Prepare FineWeb-Edu dataset in Parquet format for BOS-aligned dataloader.

Downloads FineWeb-Edu from HuggingFace and converts it to Parquet shards using
nanochat's exact approach: character-based sharding with complete row groups.

Output Format:
- shard_00000.parquet, shard_00001.parquet, ..., shard_NNNNN.parquet
- Each shard: ~250M characters (~100MB compressed)
- Row group size: 1024 (power of 2 for DDP efficiency)
- Compression: zstd level 3
- Text column with individual documents (NOT concatenated)
- Shuffle seed: 42 for reproducibility

Dataset Split:
- Training: All shards except last
- Validation: Last full shard
- Leftovers: Discarded (<0.1% of data)

Usage:
    python data/fineweb_edu/prepare_parquet.py --config sample-10BT

Requirements:
    - Disk space: ~20GB (sample-10BT), ~200GB (sample-100BT), ~700GB (sample-350BT)
    - Memory: ~16GB RAM minimum
"""

import argparse
import os
import shutil
import sys
import time

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import load_dataset
from tqdm import tqdm

# Add project src directory to path (if needed for imports)
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "src"))

# ==============================================================================
# NANOCHAT SHARDING APPROACH - KEY CONCEPTS
# ==============================================================================
#
# This script replicates nanochat's exact Parquet sharding strategy for
# efficient distributed training. Understanding these design decisions helps
# debug issues and optimize performance.
#
# 1. CHARACTER-BASED SHARDING (Why not document count?)
#    ────────────────────────────────────────────────────
#    Documents vary wildly in length:
#    - Short: "Hello world" (50 chars)
#    - Long: Academic papers (50,000+ chars)
#
#    If we use fixed document count (e.g., 100K docs/shard):
#    - Some shards could be 50MB, others 500MB
#    - Memory usage unpredictable during training
#    - Disk space estimation difficult
#
#    Character-based (~250M chars/shard) gives:
#    - Consistent ~100MB compressed size per shard
#    - Predictable memory usage for dataloader
#    - Easy disk space calculation: N shards × 100MB
#
# 2. ROW GROUP SIZE = 1024 (Why power of 2?)
#    ─────────────────────────────────────────
#    Parquet organizes rows into "row groups" for efficient columnar storage.
#    Using 1024 (2^10) plays nicely with distributed training:
#
#    - DDP world sizes are typically powers of 2: 1, 2, 4, 8, 16, 32, 64
#    - Each worker can read aligned row groups without splitting
#    - Example with 8 GPUs: Each GPU reads 1024/8 = 128 rows at a time
#
#    Nanochat uses 1024 (HuggingFace defaults to 1000, which is not a power of 2)
#
# 3. DUAL CONDITION FOR SHARD BOUNDARY (Why both checks?)
#    ──────────────────────────────────────────────────────
#    Shard is written when BOTH conditions are met:
#    a) Character count >= 250M
#    b) Document count is multiple of 1024
#
#    Why both?
#    - Condition (a): Ensures consistent shard size
#    - Condition (b): Ensures last row group is complete (not partial)
#
#    Partial row groups cause issues in distributed reading because
#    workers may try to read beyond the actual data.
#
#    What happens if conditions are NOT met?
#    ─────────────────────────────────────────
#    The loop continues accumulating documents. Let's see scenarios:
#
#    Scenario 1: Early stage - both conditions false
#      chars=180M, docs=512 → (a)=FALSE, (b)=FALSE → Keep accumulating
#
#    Scenario 2: Enough chars but doc count not aligned
#      chars=251M, docs=1023 → (a)=TRUE, (b)=FALSE
#      → Keep going! Add 1 more doc to reach 1024
#      → Final: chars=253M, docs=1024 → Write shard
#
#    Scenario 3: Docs aligned but not enough chars (COMMON)
#      chars=180M, docs=1024 → (a)=FALSE, (b)=TRUE → Keep accumulating
#      → Will continue adding docs until we hit 250M+ chars
#
#    KEY INSIGHT: Shards can exceed 250M chars!
#    ──────────────────────────────────────────
#    If we have 251M chars at 1023 docs, we'll add more docs until
#    we hit 1024. This means shards are typically 250M-260M chars.
#
#    The extra chars are acceptable because:
#    - Still results in ~100-105MB shards (predictable)
#    - Worth it to have complete row groups (better for DDP)
#
# 4. ZSTD COMPRESSION LEVEL 3 (Why not snappy?)
#    ─────────────────────────────────────────────
#    - Snappy: Fast but larger files (~150MB/shard)
#    - Zstd level 3: 30-40% better compression (~100MB/shard)
#    - Still fast enough for on-the-fly decompression during training
#    - Saves significant disk space for large datasets (10BT → 2TB saved!)
#
# 5. DOCUMENTS NOT CONCATENATED (Why separate rows?)
#    ──────────────────────────────────────────────────
#    Each document is a separate row in the Parquet table:
#
#    Row 0: "Document 1 text..."
#    Row 1: "Document 2 text..."
#    Row 2: "Document 3 text..."
#
#    NOT: "Document 1...Document 2...Document 3..." (concatenated)
#
#    Why?
#    - Dataloader uses best-fit packing algorithm to minimize cropping
#    - Needs to access individual documents to pack efficiently
#    - Can prepend BOS token to each document during tokenization
#
# 6. TEMP DIRECTORY OPTIMIZATION (Why not write directly?)
#    ────────────────────────────────────────────────────────
#    Network filesystems (/sensei-fs) are 10-100x slower than local SSD
#
#    Strategy:
#    1. Write all shards to local SSD (/mnt/localssd) - Fast!
#    2. Bulk move to network FS after completion
#
#    Saves hours of processing time for large datasets!
#
# 7. TRAIN/VAL SPLIT (Why no explicit split?)
#    ───────────────────────────────────────────
#    Nanochat doesn't create an explicit train/val split. Instead:
#
#    1. Shuffle entire dataset with fixed seed (42)
#    2. Process documents into shards sequentially
#    3. Discard leftover documents that don't meet both conditions (<0.1% of data)
#    4. Dataloader uses convention: all shards except last = train, last = val
#
#    Why discard leftovers instead of writing them?
#    - Maintains strict shard size consistency (~100MB each)
#    - All shards have complete row groups (1024 docs)
#    - For huge datasets (100BT), losing <0.1% is negligible
#    - Validation shard is full-sized like training shards
#
#    The last FULL shard becomes validation by dataloader convention.
#
# ==============================================================================

# Configuration
FINEWEB_CONFIG = (
    "sample-10BT"  # Options: sample-10BT, sample-100BT, sample-350BT, default
)
CHARS_PER_SHARD = 250_000_000  # ~100MB compressed shards
ROW_GROUP_SIZE = 1024  # Power of 2 for DDP efficiency
NUM_PROC = 8  # Parallel workers for dataset loading


def create_parquet_shards(
    dataset, output_dir, split_name, chars_per_shard=CHARS_PER_SHARD
):
    """
    Convert HuggingFace dataset to Parquet shards using nanochat's character-based approach.

    Key design decisions (see top of file for detailed explanations):
    - Character-based sharding for consistent ~100MB shard sizes
    - Power-of-2 row groups (1024) for efficient DDP reading
    - Dual condition (chars + doc alignment) ensures complete row groups
    - Leftovers (<0.1%) discarded to maintain strict shard consistency
    - Last full shard becomes validation set by dataloader convention

    Args:
        dataset: HuggingFace Dataset with 'text' column
        output_dir: Directory to save Parquet files
        split_name: Description for progress display (e.g., 'all', 'train')
        chars_per_shard: Target characters per shard (~250M for ~100MB)

    Returns:
        list: Paths to created Parquet files
    """
    print(f"\n📦 Creating shards from {split_name} split (nanochat-style)")
    print(
        f"   Documents: {len(dataset):,} | Target chars/shard: {chars_per_shard:,} | Row group size: {ROW_GROUP_SIZE}"
    )

    os.makedirs(output_dir, exist_ok=True)
    shard_paths = []

    # Shard accumulation state
    shard_docs = []
    shard_index = 0
    shard_characters = 0

    # Progress tracking
    total_docs_processed = 0
    total_time_spent = 0
    t0 = time.time()

    for idx in tqdm(range(len(dataset)), desc="Creating shards"):
        text = dataset[idx]["text"]
        shard_docs.append(text)
        shard_characters += len(text)

        # Dual condition for writing shard (see explanation at top of file)
        # Both must be True to ensure consistent size AND complete row groups
        collected_enough_chars = shard_characters >= chars_per_shard
        docs_aligned_to_row_groups = len(shard_docs) % ROW_GROUP_SIZE == 0

        if collected_enough_chars and docs_aligned_to_row_groups:
            shard_filename = f"shard_{shard_index:05d}.parquet"
            shard_path = os.path.join(output_dir, shard_filename)

            # Write Parquet: each doc is a separate row (not concatenated)
            shard_table = pa.Table.from_pydict({"text": shard_docs})
            pq.write_table(
                shard_table,
                shard_path,
                row_group_size=ROW_GROUP_SIZE,
                use_dictionary=False,  # Not useful for varied text
                compression="zstd",
                compression_level=3,  # Balance between speed and compression
                write_statistics=False,  # Not useful for text
            )

            # Progress metrics and ETA
            t1 = time.time()
            dt = t1 - t0
            t0 = t1
            total_docs_processed += len(shard_docs)
            total_time_spent += dt

            size_mb = os.path.getsize(shard_path) / (1024 * 1024)
            remaining_docs = len(dataset) - total_docs_processed
            eta_hours = (
                remaining_docs * total_time_spent / total_docs_processed
            ) / 3600

            print(
                f"   Shard {shard_index:05d}: {len(shard_docs):,} docs | "
                f"{shard_characters:,} chars | {size_mb:.2f} MB | "
                f"{dt:.2f}s | ETA: {eta_hours:.2f}h"
            )

            # Reset for next shard
            shard_paths.append(shard_path)
            shard_docs = []
            shard_characters = 0
            shard_index += 1

    # Discard leftovers that don't meet dual conditions (nanochat convention)
    if shard_docs:
        print(
            f"\n   ⚠️  Discarded {len(shard_docs):,} leftover docs ({shard_characters:,} chars)"
        )
        print(
            f"       Reason: Didn't meet both conditions (chars >= {chars_per_shard:,} AND docs % {ROW_GROUP_SIZE} == 0)"
        )

    total_size_gb = sum(os.path.getsize(p) for p in shard_paths) / (1024**3)
    print(
        f"\n✅ Created {len(shard_paths)} shards | Total size: {total_size_gb:.2f} GB"
    )

    return shard_paths


def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Prepare FineWeb-Edu in Parquet format"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="/sensei-fs/users/divgoyal/fineweb_edu_parquet",
        help="Final directory to save Parquet shards",
    )
    parser.add_argument(
        "--tmp_dir",
        type=str,
        default="/mnt/localssd/VibeNanoChat/data/tmp_shards",
        help="Temporary local directory for creating shards (faster for network FS)",
    )
    parser.add_argument(
        "--cache_dir",
        type=str,
        default="/sensei-fs/users/divgoyal/fineweb_edu/hf_cache",
        help="HuggingFace cache directory",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=FINEWEB_CONFIG,
        choices=["sample-10BT", "sample-100BT", "sample-350BT", "default"],
        help="FineWeb-Edu config to use",
    )
    parser.add_argument(
        "--chars_per_shard",
        type=int,
        default=CHARS_PER_SHARD,
        help="Target characters per Parquet shard (~250M for ~100MB)",
    )
    args = parser.parse_args()

    print("=" * 80)
    print("📚 FineWeb-Edu Parquet Preparation (nanochat-style)")
    print("=" * 80)
    print(f"\nConfig: {args.config} | Chars/shard: {args.chars_per_shard:,}")
    print(f"Temp dir: {args.tmp_dir}")
    print(f"Output dir: {args.output_dir}\n")

    # Step 1: Load dataset from HuggingFace (cached if already downloaded)
    print(f"📥 Loading FineWeb-Edu (config: {args.config})...")
    print("   This may take a while for larger subsets...\n")

    dataset = load_dataset(
        "HuggingFaceFW/fineweb-edu",
        name=args.config,
        num_proc=NUM_PROC,
        cache_dir=args.cache_dir,
    )

    # Step 2: Shuffle with fixed seed (seed=42 for reproducibility)
    print("🔀 Shuffling dataset (seed=42)...")
    shuffled_dataset = dataset["train"].shuffle(seed=42)
    print(f"   Total documents: {len(shuffled_dataset):,}\n")

    # Step 3: Create shards in temp directory (local SSD is 10-100x faster than network FS)
    os.makedirs(args.tmp_dir, exist_ok=True)
    print(f"📁 Creating shards in temp directory: {args.tmp_dir}")
    print("   (Will move to final location after completion)\n")

    all_shards = create_parquet_shards(
        shuffled_dataset,
        args.tmp_dir,
        "all",
        args.chars_per_shard,
    )

    # Step 4: Move shards to final destination (atomic move handles cross-filesystem)
    print("\n" + "=" * 80)
    print("📦 Moving shards to final destination")
    print("=" * 80)
    os.makedirs(args.output_dir, exist_ok=True)

    for shard_path in tqdm(all_shards, desc="Moving shards"):
        filename = os.path.basename(shard_path)
        dest_path = os.path.join(args.output_dir, filename)
        shutil.move(shard_path, dest_path)

    print(f"✅ All shards moved to {args.output_dir}")

    # Clean up temp directory if empty
    if os.path.exists(args.tmp_dir) and not os.listdir(args.tmp_dir):
        os.rmdir(args.tmp_dir)
        print("✅ Cleaned up temporary directory")

    # Summary and usage instructions
    print("\n" + "=" * 80)
    print("✅ PREPARATION COMPLETE")
    print("=" * 80)
    print(f"\n📂 Output directory: {args.output_dir}")
    print("\n📊 Dataset Summary:")
    print(f"   Total shards: {len(all_shards)}")

    # Handle edge case: single shard means only validation, no training data
    if len(all_shards) == 1:
        print("   Training shards: None (only 1 shard available)")
        print("   Validation shard: shard_00000 (only shard)")
    else:
        print(
            f"   Training shards: shard_00000 to shard_{len(all_shards)-2:05d} ({len(all_shards)-1} shards)"
        )
        print(f"   Validation shard: shard_{len(all_shards)-1:05d} (last full shard)")

    total_docs = len(shuffled_dataset)
    total_size = sum(
        os.path.getsize(os.path.join(args.output_dir, f"shard_{i:05d}.parquet"))
        for i in range(len(all_shards))
    )
    print(f"\n   Total documents: {total_docs:,}")
    print(f"   Total size: {total_size / (1024**3):.2f} GB")

    print("\n📝 Format Details:")
    print(f"   - Target: ~{args.chars_per_shard:,} chars/shard (~100MB compressed)")
    print(f"   - Row groups: {ROW_GROUP_SIZE} docs/group (power of 2 for DDP)")
    print("   - Compression: zstd level 3")
    print("   - Shuffle seed: 42")
    print("   - Storage: Raw documents in 'text' column (NOT concatenated)")

    print("\n🚀 Usage in Training:")
    print("   1. Set in config.py:")
    print(f"      data_dir_pretrain_parquet = '{args.output_dir}'")
    print("\n   2. Dataloader behavior:")
    print("      • Reads all shards except last for training")
    print("      • Uses last shard for validation")
    print("      • Tokenizes on-the-fly (not pre-tokenized)")
    print("      • Prepends BOS token to each document")
    print("      • Packs documents with best-fit algorithm (~35% cropping)")
    print("      • Reads row groups in parallel across DDP ranks")


if __name__ == "__main__":
    main()
