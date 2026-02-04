"""
PyTorch-native BOS-aligned dataloader with best-fit packing.

This dataloader leverages PyTorch's DataLoader infrastructure:
- Uses torch.utils.data.IterableDataset for document streaming
- Leverages DataLoader's num_workers for parallel I/O and tokenization
- Built-in prefetching via prefetch_factor
- Automatic pin_memory and non_blocking transfers
- Custom collate_fn for best-fit packing algorithm

Key features:
- Simpler code (no manual threading)
- Better multiprocessing (leverages PyTorch's worker pool)
- Automatic worker lifecycle management
- Full integration with PyTorch ecosystem
- Reliable shutdown (no thread leaks)

Architecture:
    [PyTorch Worker Pool (num_workers=4)]
        Parquet Files → Read → Tokenize → Document Queue (prefetch_factor=2)
                                                    ↓
    [Main Thread]                                   ↓
        Custom collate_fn: Buffer + Best-Fit → Batches (B, T)
"""

import glob
import os
from typing import Iterator, List, Optional, Tuple

import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, IterableDataset

from gpt_2.utils import get_custom_tokenizer


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
        # Get worker info for DataLoader multiprocessing
        worker_info = torch.utils.data.get_worker_info()

        if worker_info is None:
            # Single-process mode
            worker_id = 0
            num_workers = 1
        else:
            # Multi-process mode: each worker gets different data
            worker_id = worker_info.id
            num_workers = worker_info.num_workers

        # Total sharding: DDP_world_size * num_workers
        # Example: 8 GPUs, 4 workers = 32 total shards
        total_shards = self.ddp_world_size * num_workers
        shard_id = self.ddp_rank * num_workers + worker_id

        # Infinite loop for multi-epoch training
        epoch = 1
        while True:
            for shard_idx, filepath in enumerate(self.parquet_paths):
                parquet_file = pq.ParquetFile(filepath)

                # Each shard reads different row groups
                # Example: With 32 total shards and 128 row groups:
                #   - Shard 0: row groups 0, 32, 64, 96
                #   - Shard 1: row groups 1, 33, 65, 97
                #   - etc.
                for row_group_idx in range(
                    shard_id, parquet_file.num_row_groups, total_shards
                ):
                    row_group = parquet_file.read_row_group(row_group_idx)
                    texts = row_group.column("text").to_pylist()

                    # Tokenize each document with BOS prepended
                    for document_text in texts:
                        token_ids = self.tokenizer.encode(
                            f"<|bos|>{document_text}", allowed_special="all"
                        )
                        if len(token_ids) > 0:  # Skip empty documents
                            yield token_ids

            epoch += 1


class BestFitCollator:
    """
    Custom collate function that packs documents using best-fit algorithm.

    Maintains a document buffer and uses best-fit cropping to minimize waste.
    This is the core packing logic from the original implementation.

    Attributes:
        batch_size: Number of sequences per batch (B)
        block_size: Length of each sequence (T)
        buffer_size: Number of documents to buffer for best-fit
        device: Target device ('cuda' or 'cpu')
        doc_buffer: Buffer of tokenized documents
        stats_*: Statistics for monitoring packing efficiency
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
        self.doc_buffer: List[List[int]] = []

        # Statistics tracking
        self.stats_total_tokens = 0
        self.stats_cropped_tokens = 0
        self.stats_skipped_documents = 0  # Documents skipped due to full buffer

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
        # Only refill buffer when it drops below 50% capacity
        # This prevents accumulating too many documents and then truncating (losing data)
        # When buffer is full, we skip the incoming documents rather than adding and truncating
        if len(self.doc_buffer) < self.buffer_size // 2:
            self.doc_buffer.extend(batch_of_documents)
        else:
            # Buffer is full - skip these documents to avoid overflow and truncation
            self.stats_skipped_documents += len(batch_of_documents)

        # If buffer too small, we can't pack a batch yet
        # This should only happen at the very start
        if len(self.doc_buffer) < self.batch_size:
            # Return empty batch (training loop should handle this gracefully)
            # Or we could block until buffer fills, but that breaks PyTorch's flow
            raise RuntimeError(
                f"Buffer too small for packing: {len(self.doc_buffer)} docs "
                f"< {self.batch_size} required. Increase buffer_size or "
                f"prefetch_factor in DataLoader."
            )

        # Pack documents into batch using best-fit algorithm
        for row_idx in range(self.batch_size):
            current_position = 0

            while current_position < self.block_size + 1:
                remaining_tokens = self.block_size + 1 - current_position

                # BEST-FIT: Find largest document that fits
                best_fit_idx = -1
                best_fit_length = 0
                for buffer_idx, token_ids in enumerate(self.doc_buffer):
                    document_length = len(token_ids)
                    if (
                        document_length <= remaining_tokens
                        and document_length > best_fit_length
                    ):
                        best_fit_idx = buffer_idx
                        best_fit_length = document_length

                if best_fit_idx >= 0:
                    # Found a document that fits entirely
                    token_ids = self.doc_buffer.pop(best_fit_idx)
                    document_length = len(token_ids)
                    self.row_buffer[
                        row_idx, current_position : current_position + document_length
                    ] = torch.tensor(token_ids, dtype=torch.long)
                    current_position += document_length
                    self.stats_total_tokens += document_length
                else:
                    # No document fits, crop the shortest
                    if len(self.doc_buffer) == 0:
                        raise RuntimeError(
                            "Document buffer exhausted during packing. "
                            "Increase buffer_size or prefetch_factor."
                        )

                    shortest_doc_idx = min(
                        range(len(self.doc_buffer)),
                        key=lambda idx: len(self.doc_buffer[idx]),
                    )
                    token_ids = self.doc_buffer.pop(shortest_doc_idx)
                    cropped_tokens = token_ids[:remaining_tokens]
                    self.row_buffer[
                        row_idx, current_position : current_position + remaining_tokens
                    ] = torch.tensor(cropped_tokens, dtype=torch.long)
                    self.stats_total_tokens += remaining_tokens
                    self.stats_cropped_tokens += len(token_ids) - remaining_tokens
                    current_position += remaining_tokens

        # Split into inputs and targets
        inputs = self.row_buffer[:, :-1].clone()  # (B, T)
        targets = self.row_buffer[:, 1:].clone()  # (B, T)

        # Return CPU tensors - training loop will handle device transfer
        # (Cannot move to CUDA in DataLoader worker processes)
        return inputs, targets

    def get_stats(self) -> dict:
        """Get packing efficiency statistics."""
        crop_pct = 100.0 * self.stats_cropped_tokens / max(1, self.stats_total_tokens)
        return {
            "total_tokens": self.stats_total_tokens,
            "cropped_tokens": self.stats_cropped_tokens,
            "crop_percentage": crop_pct,
            "buffer_size": len(self.doc_buffer),
            "skipped_documents": self.stats_skipped_documents,
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

        # Create dataset
        dataset = ParquetDocumentDataset(
            split_parquet_files,
            tokenizer,
            ddp_rank,
            ddp_world_size,
            split,
        )

        # Create collator for best-fit packing
        self.collator = BestFitCollator(batch_size, block_size, buffer_size, device)

        # PyTorch DataLoader requires prefetch_factor=None when num_workers=0
        # Also, persistent_workers must be False when num_workers=0
        if num_workers == 0:
            prefetch_factor = None
            persistent_workers = False

        # Create PyTorch DataLoader with all optimizations
        # Key features:
        # - num_workers: Parallel I/O and tokenization
        # - prefetch_factor: Each worker prefetches N batches (None for single-process)
        # - pin_memory: Pinned CPU memory for faster GPU transfer
        # - persistent_workers: Avoid worker startup overhead between epochs
        #
        # DataLoader batch_size: Number of documents to feed to collate_fn at once
        # For best-fit to work well, we need enough documents in the buffer
        # Use max(buffer_size // 4, batch_size * 4) to ensure enough documents
        dataloader_batch_size = max(buffer_size // 4, batch_size * 4)

        # Enable pin_memory for faster CPU-to-GPU transfers when using CUDA
        use_pin_memory = device == "cuda" and num_workers > 0

        self.dataloader = DataLoader(
            dataset,
            batch_size=dataloader_batch_size,  # Documents per collate call
            collate_fn=self.collator,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=use_pin_memory,  # Pin memory for faster GPU transfer
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
