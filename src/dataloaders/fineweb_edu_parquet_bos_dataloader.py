"""
PyTorch-native BOS-aligned dataloader with best-fit packing.

=== HIGH-LEVEL OVERVIEW ===

Problem: Train LLMs on massive datasets (billions of documents) efficiently
- Documents have variable length (50 to 5000 tokens)
- We need fixed-size batches (B, T) for GPU training
- Want to minimize wasted tokens (cropping/padding waste)
- Need to scale across multiple GPUs (DDP) without data duplication

Solution: Three-stage pipeline leveraging PyTorch DataLoader:

Stage 1: DISTRIBUTED STREAMING (ParquetDocumentDataset)
  - All workers read ALL parquet files (no file-level sharding)
  - Within each file, use strided row group reads (prevents duplication)
  - Example: 4 GPUs × 4 workers = 16 shards
    → Shard 0 reads row groups [0, 16, 32, 48, ...]
    → Shard 1 reads row groups [1, 17, 33, 49, ...]
  - Each document gets BOS token prepended: "<|bos|>document text"

Stage 2: BEST-FIT PACKING (BestFitCollator)
  - Maintain buffer of ~300 variable-length documents (SortedList)
  - Pack documents into fixed sequences (B, T+1) using best-fit algorithm
  - Best-fit: O(log n) binary search for largest document that fits
  - When no document fits, crop the shortest one (minimize waste)
  - Typical waste: ~37% of tokens cropped at document boundaries

Stage 3: PYTORCH DATALOADER (FinewebEduParquetBOSDataloader)
  - num_workers=4: parallel I/O, tokenization in worker processes
  - prefetch_factor=2: each worker prefetches 2 batches (hides latency)
  - pin_memory=True: async CPU→GPU transfers (faster than sync copies)
  - persistent_workers=True: keep workers alive across epochs

Architecture:
    [4 GPUs × 4 Workers = 16 Parallel Streams]
        Parquet Files → Read Row Groups → Tokenize → Document Stream
                                                            ↓
    [Main Process on each GPU]                              ↓
        BestFitCollator: Buffer 1000 docs → Pack into (B, T) → Training

Key features:
✓ No data duplication across GPUs/workers (perfect sharding)
✓ Efficient packing (only ~35% token waste, vs ~50% for padding)
✓ Scales to any number of GPUs/workers
✓ Leverages PyTorch's battle-tested DataLoader (no custom threading)
"""

import glob
import os
from typing import Iterator, List, Optional, Tuple

import pyarrow.parquet as pq
import torch
from sortedcontainers import SortedList
from torch.utils.data import DataLoader, IterableDataset

from gpt_2.utils import get_custom_tokenizer


class SortedDocument:
    """
    Wrapper for documents that enables sorting by length.

    Stores both the document and its length for efficient comparison.
    Implements comparison operators so SortedList can maintain sort order.
    """

    __slots__ = ("tokens", "length")

    def __init__(self, tokens: List[int]):
        self.tokens = tokens
        self.length = len(tokens)

    def __lt__(self, other):
        if isinstance(other, SortedDocument):
            return self.length < other.length
        return self.length < other  # Allow comparison with int

    def __le__(self, other):
        if isinstance(other, SortedDocument):
            return self.length <= other.length
        return self.length <= other

    def __gt__(self, other):
        if isinstance(other, SortedDocument):
            return self.length > other.length
        return self.length > other

    def __ge__(self, other):
        if isinstance(other, SortedDocument):
            return self.length >= other.length
        return self.length >= other

    def __eq__(self, other):
        if isinstance(other, SortedDocument):
            return self.length == other.length
        return self.length == other

    def __len__(self):
        return self.length


def list_parquet_files(data_dir: str) -> List[str]:
    """List all Parquet files in directory, sorted by name."""
    parquet_files = sorted(
        [
            filepath
            for filepath in glob.glob(os.path.join(data_dir, "*.parquet"))
            if not filepath.endswith(".tmp")
        ]
    )
    return parquet_files


class ParquetDocumentDataset(IterableDataset):
    """
    PyTorch IterableDataset that streams tokenized documents from Parquet files.

    Each worker handles a subset of row groups (DDP + DataLoader sharding).
    Documents are tokenized with BOS token prepended on-the-fly.

    Attributes:
        parquet_paths: List of parquet file paths
        tokenizer: Tokenizer with BOS support
        ddp_rank: DDP process rank
        ddp_world_size: Total DDP processes
        split: 'train' or 'val'
    """

    def __init__(
        self,
        parquet_paths: List[str],
        tokenizer,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        split: str = "train",
    ):
        super().__init__()
        self.parquet_paths = parquet_paths
        self.tokenizer = tokenizer
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.split = split

    def __iter__(self) -> Iterator[List[int]]:
        """
        Yield tokenized documents with BOS token.

        Handles both DDP sharding and DataLoader worker sharding automatically.
        """
        # ---- STEP 1: Figure out which worker we are ----
        # PyTorch DataLoader can spawn multiple worker processes (num_workers=4)
        # Each worker needs to know its ID so it can claim different data
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process mode (num_workers=0): we are the only worker
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process mode: PyTorch tells us our worker_id (0, 1, 2, 3...)
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # ---- STEP 2: Calculate our unique shard ID ----
        # We have two levels of parallelism:
        #   1. DDP parallelism: multiple GPUs (e.g., 4 GPUs)
        #   2. DataLoader parallelism: multiple workers per GPU (e.g., 4 workers)
        # Total parallelism = 4 GPUs × 4 workers = 16 concurrent data streams
        global_num_workers = self.ddp_world_size * num_workers

        # Only print from worker 0 on rank 0 to avoid spam (would print 16 times otherwise!)
        if self.ddp_rank == 0 and worker_id == 0:
            print(
                f"Global num workers: {global_num_workers}, DDP World Size: {self.ddp_world_size}, Num Workers per GPU: {num_workers}"
            )

        # Give each worker a unique shard_id from 0 to (global_num_workers-1)
        # Formula ensures no overlap: rank 0's workers get 0-3, rank 1's get 4-7, etc.
        shard_id = self.ddp_rank * num_workers + worker_id
        # Example with 4 GPUs and 4 workers (16 total shards):
        # GPU 0, worker 0: shard_id = 0*4 + 0 = 0
        # GPU 0, worker 1: shard_id = 0*4 + 1 = 1
        # GPU 0, worker 2: shard_id = 0*4 + 2 = 2
        # GPU 0, worker 3: shard_id = 0*4 + 3 = 3
        # GPU 1, worker 0: shard_id = 1*4 + 0 = 4
        # GPU 1, worker 1: shard_id = 1*4 + 1 = 5
        # GPU 2, worker 0: shard_id = 2*4 + 0 = 8
        # GPU 3, worker 0: shard_id = 3*4 + 0 = 12
        # ... (ensures no data duplication across all workers)

        # ---- STEP 3: Stream data forever (multi-epoch training) ----
        # We loop infinitely because training might need multiple epochs
        # The training loop will stop us when it's done
        epoch = 1
        while True:
            # ---- STEP 4: All workers read ALL parquet files ----
            # Key insight: we don't split files across GPUs, we split ROW GROUPS
            # This ensures better load balancing (no file could be too small/large)
            for shard_idx, filepath in enumerate(self.parquet_paths):
                parquet_file = pq.ParquetFile(filepath)

                # ---- STEP 5: Use strided indexing to claim our row groups ----
                # Parquet files contain "row groups" (chunks of ~1024 documents each)
                # We use strided indexing: start at shard_id, skip by global_num_workers
                # Example: 16 global workers, 64 row groups in this file:
                #   - Shard 0: reads row groups 0, 16, 32, 48 (every 16th starting at 0)
                #   - Shard 1: reads row groups 1, 17, 33, 49 (every 16th starting at 1)
                #   - Shard 15: reads row groups 15, 31, 47, 63 (every 16th starting at 15)
                # Result: each shard gets ~4 row groups, perfectly non-overlapping
                for row_group_idx in range(
                    shard_id, parquet_file.num_row_groups, global_num_workers
                ):
                    # Read this row group (contains ~1024 documents as a batch)
                    row_group = parquet_file.read_row_group(row_group_idx)
                    texts = row_group.column("text").to_pylist()

                    # ---- STEP 6: Tokenize each document ----
                    # Prepend BOS token (<|bos|>) so model knows where document starts
                    # This is crucial for document boundary awareness during training
                    for document_text in texts:
                        token_ids = self.tokenizer.encode(
                            f"<|bos|>{document_text}", allowed_special="all"
                        )
                        if (
                            len(token_ids) > 0
                        ):  # Skip empty documents (rare but possible)
                            yield token_ids  # Return to collator for packing into batches

            epoch += 1  # Just for bookkeeping, infinite loop continues


class BestFitCollator:
    """
    Custom collate function that packs documents using best-fit algorithm.

    Maintains a document buffer (SortedList) and uses best-fit cropping to minimize waste.
    Uses binary search for O(log n) best-fit lookup instead of O(n) linear scan.

    Attributes:
        batch_size: Number of sequences per batch (B)
        block_size: Length of each sequence (T)
        buffer_size: Number of documents to buffer for best-fit
        device: Target device ('cuda' or 'cpu')
        doc_buffer: SortedList of tokenized documents (sorted by length)
        stats_*: Statistics for monitoring packing efficiency

    Complexity:
        - Insert document: O(log n)
        - Find best-fit: O(log n) via binary search
        - Find shortest: O(1) (always index 0)
        - Remove document: O(log n)
    """

    def __init__(
        self,
        batch_size: int,
        block_size: int,
        buffer_size: int = 1000,
        device: str = "cuda",
    ):
        self.batch_size = batch_size
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.device = device

        # Document buffer for best-fit algorithm
        # Use SortedList for O(log n) binary search instead of O(n) linear scan
        # Documents are wrapped in SortedDocument for efficient length-based sorting
        self.doc_buffer = SortedList()

        # Statistics tracking
        self.stats_total_tokens = 0
        self.stats_cropped_tokens = 0

        # Pre-allocate buffers (same strategy as v1)
        self.row_buffer = torch.empty((batch_size, block_size + 1), dtype=torch.long)

    def __call__(
        self, batch_of_documents: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Pack a batch of documents into fixed-size sequences using best-fit.

        Args:
            batch_of_documents: List of tokenized documents from DataLoader

        Returns:
            Tuple of (inputs, targets) tensors of shape (batch_size, block_size)
        """
        # ---- STEP 1: Manage document buffer ----
        # Accept incoming documents - packing provides natural backpressure
        # The synchronous collator blocks DataLoader workers while packing
        # Wrap documents in SortedDocument for efficient length-based sorting
        self.doc_buffer.update(SortedDocument(doc) for doc in batch_of_documents)

        # Safety check: prevent pathological buffer growth
        # In high-parallelism setups, buffer can grow during startup bursts
        # With buffer_size=300, allow 10x headroom = 3000 docs max
        # This is plenty for: 8 workers × 2 prefetch × 48 docs = 768 docs in-flight
        max_buffer_size = self.buffer_size * 10
        if len(self.doc_buffer) > max_buffer_size:
            raise RuntimeError(
                f"Buffer overflow: {len(self.doc_buffer)} > {max_buffer_size}. "
                f"Buffer stats - arrived: +{len(batch_of_documents)} docs, "
                f"total buffered: {len(self.doc_buffer)} docs. "
                f"This indicates packing cannot keep up with worker production. "
                f"Try reducing num_workers or dataloader_batch_size."
            )

        # Sanity check: need at least batch_size documents to pack a batch
        # This should only trigger at startup before buffer fills
        if len(self.doc_buffer) < self.batch_size:
            raise RuntimeError(
                f"Buffer too small for packing: {len(self.doc_buffer)} docs "
                f"< {self.batch_size} required. Increase buffer_size or "
                f"prefetch_factor in DataLoader."
            )

        # ---- STEP 2: Pack documents into fixed-size sequences ----
        # Goal: fill each row (sequence) of shape (block_size+1,) with documents
        # We pack multiple documents into one sequence to maximize token utilization
        # Example: block_size=2048, we might pack 5 documents: [500, 300, 800, 200, 248]
        for row_idx in range(self.batch_size):
            current_position = 0  # Track how many tokens we've packed into this row

            # Keep packing documents until row is full
            while current_position < self.block_size + 1:
                remaining_tokens = self.block_size + 1 - current_position

                # ---- BEST-FIT ALGORITHM (O(log n) with SortedList) ----
                # Search buffer for the LARGEST document that still fits
                # SortedList is sorted by document length, so use binary search
                # bisect_right(remaining_tokens) finds where docs > remaining_tokens start
                # The document at position (idx-1) is the largest that fits
                best_fit_idx = self.doc_buffer.bisect_right(remaining_tokens) - 1

                if best_fit_idx >= 0 and best_fit_idx < len(self.doc_buffer):
                    # ---- CASE 1: Found a document that fits perfectly (or close) ----
                    sorted_doc = self.doc_buffer.pop(best_fit_idx)
                    token_ids = sorted_doc.tokens
                    document_length = sorted_doc.length
                    # Copy entire document into row
                    self.row_buffer[
                        row_idx, current_position : current_position + document_length
                    ] = torch.tensor(token_ids, dtype=torch.long)
                    current_position += document_length
                    self.stats_total_tokens += document_length
                else:
                    # ---- CASE 2: No document fits, must crop one ----
                    # We're left with a small gap (e.g., 50 tokens) but all docs are bigger
                    # Strategy: crop the SHORTEST document (minimizes wasted tokens)
                    if len(self.doc_buffer) == 0:
                        raise RuntimeError(
                            "Document buffer exhausted during packing. "
                            "Increase buffer_size or prefetch_factor."
                        )

                    # Find shortest document in buffer
                    # SortedList is sorted by length, so shortest is at index 0
                    shortest_doc_idx = 0
                    sorted_doc = self.doc_buffer.pop(shortest_doc_idx)
                    token_ids = sorted_doc.tokens
                    # Crop to fit remaining space
                    cropped_tokens = token_ids[:remaining_tokens]
                    self.row_buffer[
                        row_idx, current_position : current_position + remaining_tokens
                    ] = torch.tensor(cropped_tokens, dtype=torch.long)
                    self.stats_total_tokens += remaining_tokens
                    # Track waste: tokens from this doc that we threw away
                    self.stats_cropped_tokens += len(token_ids) - remaining_tokens
                    current_position += remaining_tokens

        # ---- STEP 3: Create input/target pairs for language modeling ----
        # Standard autoregressive setup: predict token t+1 given tokens 0...t
        # Shape: row_buffer is (B, T+1), split into inputs (B, T) and targets (B, T)
        inputs = self.row_buffer[:, :-1].clone()  # (B, T) - tokens 0 to T-1
        targets = self.row_buffer[:, 1:].clone()  # (B, T) - tokens 1 to T

        # Return CPU tensors - training loop will move to GPU
        # (Cannot move to CUDA here: DataLoader workers can't access GPU)
        return inputs, targets

    def get_stats(self) -> dict:
        """Get packing efficiency statistics."""
        crop_pct = 100.0 * self.stats_cropped_tokens / max(1, self.stats_total_tokens)
        return {
            "total_tokens": self.stats_total_tokens,
            "cropped_tokens": self.stats_cropped_tokens,
            "crop_percentage": crop_pct,
            "buffer_size": len(self.doc_buffer),
        }


class FinewebEduParquetBOSDataloader:
    """
    PyTorch-native BOS-aligned dataloader with best-fit packing.

    Implementation using PyTorch's DataLoader infrastructure:
    - Leverages num_workers for parallel I/O and tokenization
    - Built-in prefetching and pin_memory
    - Automatic worker lifecycle management
    - Custom collate_fn for best-fit packing

    Usage:
        >>> dataloader = FinewebEduParquetBOSDataloader(
        ...     data_dir="/path/to/fineweb_edu_parquet",
        ...     batch_size=16,
        ...     block_size=1024,
        ...     num_workers=4,
        ... )
        >>> for inputs, targets in dataloader:
        ...     loss = model(inputs, targets)
    """

    def __init__(
        self,
        data_dir: str,
        batch_size: int,
        block_size: int,
        ddp_world_size: int = 1,
        ddp_rank: int = 0,
        split: str = "train",
        master_process: bool = True,
        buffer_size: int = 1000,
        device: str = "cuda",
        num_workers: int = 4,
        prefetch_factor: Optional[int] = 2,
        persistent_workers: bool = True,
    ):
        """
        Initialize PyTorch-native BOS dataloader.

        Args:
            data_dir: Directory containing parquet shards
            batch_size: Batch size (B)
            block_size: Sequence length (T)
            ddp_world_size: Number of DDP processes
            ddp_rank: Current DDP rank
            split: 'train' or 'val'
            master_process: Whether this is master process
            buffer_size: Document buffer size for best-fit
            device: Target device ('cuda' or 'cpu')
            num_workers: Number of DataLoader workers (parallel I/O)
            prefetch_factor: Batches to prefetch per worker (None=auto)
            persistent_workers: Keep workers alive between epochs
        """
        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.split = split

        # Initialize tokenizer
        tokenizer, _ = get_custom_tokenizer()
        bos_token_id = tokenizer.encode("<|bos|>", allowed_special="all")[0]

        if master_process:
            print(f"\n{'='*80}")
            print(f"📚 PyTorch-Native BOS Dataloader ({split})")
            print(f"{'='*80}")
            print(f"Data directory: {data_dir}")
            print(f"Batch size: {batch_size}")
            print(f"Block size: {block_size}")
            print(f"Buffer size: {buffer_size}")
            print(f"BOS token ID: {bos_token_id}")
            print(f"Num workers: {num_workers}")
            print(f"Prefetch factor: {prefetch_factor}")
            print(f"Persistent workers: {persistent_workers}")

        # Get parquet files
        parquet_files = list_parquet_files(data_dir)
        if len(parquet_files) == 0:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        # Split files: train gets all but last, val gets only last
        if split == "train":
            split_parquet_files = parquet_files[:-1]
        else:
            split_parquet_files = parquet_files[-1:]

        if len(split_parquet_files) == 0:
            raise ValueError(
                f"No parquet files for split='{split}'. "
                f"Need at least 2 shards (one for train, one for val)."
            )

        if master_process:
            print(f"Found {len(parquet_files)} parquet files")
            if split == "train":
                print(
                    f"Using files: {os.path.basename(parquet_files[0])} to "
                    f"{os.path.basename(parquet_files[-2])}"
                )
            else:
                print(f"Using file: {os.path.basename(parquet_files[-1])}")
            print(f"{'='*80}\n")

        # ---- Create dataset (handles streaming and sharding) ----
        dataset = ParquetDocumentDataset(
            split_parquet_files,
            tokenizer,
            ddp_rank,
            ddp_world_size,
            split,
        )

        # ---- Create collator (handles packing documents into sequences) ----
        # This is where the magic happens: variable-length docs → fixed-size batches
        self.collator = BestFitCollator(batch_size, block_size, buffer_size, device)

        # ---- Handle single-worker edge case ----
        # PyTorch requires prefetch_factor=None and persistent_workers=False when num_workers=0
        if num_workers == 0:
            prefetch_factor = None
            persistent_workers = False

        # ---- Configure DataLoader batch size ----
        # IMPORTANT: DataLoader's "batch_size" means documents per collate_fn call
        # This is NOT the model batch size! The collator packs these docs into sequences.
        #
        # Balance: enough docs for good best-fit, but not so many that buffer overflows
        # With 2 workers × 2 prefetch = 4 batches in flight per GPU
        # Total in-flight across 4 GPUs: 8 workers × 2 prefetch × batch = 16 × batch
        # Example: batch_size=16 → dataloader_batch_size=48 → 768 docs in-flight
        dataloader_batch_size = max(batch_size * 3, 48)

        # ---- Enable pin_memory for faster GPU transfers ----
        # Pinned memory allows async CPU→GPU copies (DMA transfer)
        # Only works with num_workers>0 (worker processes can use pinned memory)
        use_pin_memory = device == "cuda" and num_workers > 0

        # ---- Create PyTorch DataLoader ----
        # This ties everything together:
        #   1. Dataset: streams tokenized documents (handles DDP + worker sharding)
        #   2. Workers: parallel I/O, tokenization, prefetching (num_workers=4)
        #   3. Collator: packs documents into fixed-size batches (best-fit algorithm)
        #   4. Prefetch: each worker prefetches 2 batches ahead (hides I/O latency)
        #   5. Persistent workers: keep workers alive across epochs (avoid startup cost)
        self.dataloader = DataLoader(
            dataset,
            batch_size=dataloader_batch_size,  # Documents per collate_fn call
            collate_fn=self.collator,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=use_pin_memory,  # Faster CPU→GPU transfers
            persistent_workers=persistent_workers,
            drop_last=False,
        )

        self._iterator = None

        if master_process:
            pin_status = "✓ enabled" if use_pin_memory else "✗ disabled"
            print(f"✓ DataLoader initialized (pin_memory: {pin_status})\n")

    def __iter__(self):
        """Make dataloader iterable."""
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self):
        """Get next batch."""
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        return next(self._iterator)

    def get_stats(self) -> dict:
        """Get token packing statistics."""
        return self.collator.get_stats()
