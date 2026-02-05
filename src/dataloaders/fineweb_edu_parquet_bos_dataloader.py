"""
BOS-aligned dataloader with best-fit packing (PyTorch-native implementation).

Adapted from nanochat's tokenizing_distributed_data_loader_with_state_bos_bestfit.
Every row starts with BOS token. Documents packed using best-fit to minimize cropping.
When no document fits remaining space, crops a document to fill exactly.
100% utilization (no padding), ~35% tokens cropped at T=2048.

Algorithm (per row):
  1. From buffered docs, pick LARGEST doc that fits entirely
  2. Repeat until no doc fits
  3. When nothing fits, crop shortest doc to fill remaining space exactly

Key properties:
  - Every row starts with BOS (critical for document boundary awareness)
  - 100% utilization (no padding, every token is trained on)
  - ~35% of tokens discarded due to cropping (vs ~50% for naive padding)
  - Scales across GPUs/workers via strided row group reads (no data duplication)

PyTorch DataLoader integration:
  - Multi-worker I/O and tokenization in parallel
  - Prefetching hides latency (prefetch_factor=2)
  - pin_memory enables async CPU→GPU transfers
  - persistent_workers avoids startup overhead
"""

import functools
import glob
import os
import random
import warnings
from typing import Iterator, List, Optional, Tuple

import pyarrow.parquet as pq
import torch
from sortedcontainers import SortedList
from torch.utils.data import DataLoader, IterableDataset

from gpt_2.utils import get_custom_tokenizer


@functools.total_ordering  # auto-generates <=, >, >= from __eq__ and __lt__
class SortedDocument:
    """Document wrapper for length-based sorting in O(log n) binary search."""

    __slots__ = ("tokens", "length")  # memory optimization: no __dict__

    def __init__(self, tokens: List[int]):
        self.tokens = tokens
        self.length = len(tokens)  # cache length for O(1) comparisons

    def __lt__(self, other):
        # supports both SortedDocument and int comparisons for bisect_right(int)
        other_len = other.length if isinstance(other, SortedDocument) else other
        return self.length < other_len

    def __eq__(self, other):
        other_len = other.length if isinstance(other, SortedDocument) else other
        return self.length == other_len

    def __len__(self):
        return self.length


def list_parquet_files(data_dir: str) -> List[str]:
    """List all .parquet files, sorted by name (excludes .tmp files)."""
    pattern = os.path.join(data_dir, "*.parquet")
    return sorted(f for f in glob.glob(pattern) if not f.endswith(".tmp"))


class ParquetDocumentDataset(IterableDataset):
    """
    Distributed document streaming with perfect sharding (nanochat-style).

    All workers read all files, but use strided row group indexing (no data duplication).
    Each document is prepended with BOS token during tokenization.

    Sharding: Worker i reads row groups [i, i+stride, i+2*stride, ...]
    Ensures load balancing (no file is too small/large for one worker).
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
        """Yield tokenized documents with BOS prepended: [<|bos|>, tok1, tok2, ...]"""

        # get worker info from PyTorch (None if num_workers=0)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        # compute global shard ID: combines DDP rank and DataLoader worker ID
        # Example: 4 GPUs × 4 workers = 16 shards, GPU 2 worker 3 → shard_id = 2*4+3 = 11
        global_num_workers = self.ddp_world_size * num_workers
        shard_id = self.ddp_rank * num_workers + worker_id

        if self.ddp_rank == 0 and worker_id == 0:  # print once to avoid spam
            print(
                f"Sharding files across: {global_num_workers} total workers "
                f"({self.ddp_world_size} GPUs × {num_workers} workers/GPU)"
            )

        # shuffle file order per worker to reduce I/O contention on shared filesystems
        # each worker gets a different file order (reduces simultaneous reads of same file)
        # use shard_id as seed for deterministic but worker-specific shuffling
        rng = random.Random(shard_id)

        # infinite loop for multi-epoch training (training loop controls stopping)
        while True:
            # shuffle files each epoch using worker-specific order
            shuffled_paths = self.parquet_paths.copy()
            rng.shuffle(shuffled_paths)

            for filepath in shuffled_paths:
                # open parquet file with proper resource management
                parquet_file = None
                try:
                    parquet_file = pq.ParquetFile(filepath)

                    # strided indexing: shard_id, shard_id + stride, shard_id + 2*stride, ...
                    # ensures each row group is read by exactly one worker (no duplication)
                    for row_group_idx in range(
                        shard_id, parquet_file.num_row_groups, global_num_workers
                    ):
                        try:
                            row_group = parquet_file.read_row_group(row_group_idx)
                            texts = row_group.column("text").to_pylist()
                        except Exception as e:
                            # handle corrupted row groups gracefully - skip and continue
                            warnings.warn(
                                f"Skipping corrupted row group {row_group_idx} in {filepath}: {e}",
                                RuntimeWarning,
                            )
                            continue

                        # tokenize with BOS prepended (critical for document boundary awareness)
                        for text in texts:
                            # skip None, empty, or non-string values
                            if not text or not isinstance(text, str):
                                continue

                            try:
                                tokens = self.tokenizer.encode(
                                    f"<|bos|>{text}", allowed_special="all"
                                )

                                # validate tokenized output
                                if len(tokens) == 0:  # skip empty docs
                                    continue
                                if (
                                    len(tokens) > 1000000
                                ):  # sanity check: skip extremely long docs (likely corrupted)
                                    warnings.warn(
                                        f"Skipping document with {len(tokens)} tokens (> 1M, likely corrupted)",
                                        RuntimeWarning,
                                    )
                                    continue

                                yield tokens
                            except Exception as e:
                                # handle tokenization errors gracefully
                                warnings.warn(
                                    f"Tokenization error, skipping document: {e}",
                                    RuntimeWarning,
                                )
                                continue
                finally:
                    # ensure parquet file is properly closed (release file handles)
                    if parquet_file is not None:
                        try:
                            parquet_file.close()
                        except Exception:
                            pass  # silently ignore close errors


class BestFitCollator:
    """
    BOS-aligned best-fit packing (nanochat-style).

    Algorithm per row:
      1. Pick LARGEST doc that fits (best-fit)
      2. Repeat until no doc fits
      3. Crop shortest doc to fill gap (optimization: minimizes waste vs nanochat's arbitrary crop)

    Properties:
      - Every row starts with BOS token
      - 100% utilization (no padding)
      - ~35% token waste from cropping (inherent to BOS-aligned packing)
      - O(log n) binary search via SortedList (vs O(n) linear scan)
    """

    def __init__(
        self,
        batch_size: int,
        block_size: int,
        buffer_size: int = 1000,
        device: str = "cuda",
    ):
        # input validation
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")

        # Convert torch.device to string if needed
        if isinstance(device, torch.device):
            device = str(device)

        if not (device == "cpu" or device.startswith("cuda")):
            raise ValueError(f"device must be 'cpu' or 'cuda[:N]', got {device}")

        self.batch_size = batch_size
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.device = device

        self.doc_buffer = SortedList()  # sorted by length for O(log n) best-fit search

        # pre-allocate once (nanochat-style): row_capacity = T + 1 for autoregressive split
        self.row_buffer = torch.empty((batch_size, block_size + 1), dtype=torch.long)

    def __call__(
        self, batch_of_documents: List[List[int]]
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        BOS-aligned best-fit packing (nanochat algorithm).

        Returns: (inputs, targets, stats) where inputs[:, t] predicts targets[:, t]
                 and stats contains per-batch metrics (deltas, not cumulative)
        """
        # Per-batch counters (track deltas only, not cumulative)
        batch_total_tokens = 0
        batch_cropped_tokens = 0
        batch_processed_tokens = 0
        batch_dropped_tokens = 0
        batch_dropped_files = 0
        batch_buffer_overflows = 0
        batch_corrupted_docs = 0
        batch_empty_docs = 0

        # Filter out empty or invalid documents and track stats
        valid_documents = []
        for doc in batch_of_documents:
            if not doc or len(doc) <= 1:  # empty or only BOS token
                batch_empty_docs += 1
            else:
                valid_documents.append(doc)

        # refill buffer with valid documents (O(log n) insert per doc)
        self.doc_buffer.update(SortedDocument(doc) for doc in valid_documents)

        # Track total tokens in THIS batch only
        batch_total_tokens = sum(len(doc) for doc in valid_documents)

        # graceful degradation: if buffer grows too large, randomly drop 25% of documents
        # this prevents OOM while signaling misconfiguration via stats
        # allow large headroom (100x) since startup can cause temporary spikes
        max_buffer = self.buffer_size * 100
        if len(self.doc_buffer) > max_buffer:
            # drop 25% of documents randomly to prevent unbounded growth
            num_to_drop = len(self.doc_buffer) // 4
            dropped_indices = random.sample(range(len(self.doc_buffer)), num_to_drop)

            # track dropped documents as wastage (sum their tokens)
            dropped_tokens = sum(self.doc_buffer[i].length for i in dropped_indices)
            batch_dropped_tokens += dropped_tokens
            batch_dropped_files += num_to_drop
            batch_buffer_overflows += 1

            # remove documents in reverse order to maintain indices
            for idx in sorted(dropped_indices, reverse=True):
                self.doc_buffer.pop(idx)

            # warn every time overflow happens (helps with debugging)
            warnings.warn(
                f"Buffer overflow: dropped {num_to_drop} documents ({dropped_tokens} tokens). "
                f"Buffer: {len(self.doc_buffer)}/{max_buffer}. "
                f"Consider reducing num_workers or increasing buffer_size. "
                f"Training continues with graceful degradation.",
                RuntimeWarning,
            )

        # pack B sequences of length T+1 (nanochat row_capacity)
        for row_idx in range(self.batch_size):
            pos = 0  # current position in this row

            # Algorithm: pick largest doc that fits, repeat until no doc fits, then crop
            while pos < self.block_size + 1:
                space = self.block_size + 1 - pos

                # STEP 1: Best-fit search - find LARGEST doc that fits (nanochat algorithm)
                # O(log n) binary search: bisect_right(space) - 1 gives largest doc <= space
                best_idx = self.doc_buffer.bisect_right(space) - 1

                if 0 <= best_idx < len(self.doc_buffer):
                    # CASE 1: Found a doc that fits - pack it entirely (nanochat step 1-2)
                    doc = self.doc_buffer.pop(best_idx)
                    # efficient: direct copy from list to tensor using slice assignment
                    try:
                        self.row_buffer[row_idx, pos : pos + doc.length] = torch.tensor(
                            doc.tokens, dtype=torch.long
                        )
                        pos += doc.length
                        batch_processed_tokens += doc.length
                    except Exception as e:
                        # handle corrupted token IDs gracefully
                        warnings.warn(
                            f"Invalid tokens in document (len={doc.length}): {e}",
                            RuntimeWarning,
                        )
                        batch_corrupted_docs += 1
                        continue  # skip this doc and try next
                else:
                    # CASE 2: No doc fits - crop one to fill gap exactly (nanochat step 3)
                    # Optimization: crop shortest doc (minimizes waste vs nanochat's arbitrary crop)
                    if len(self.doc_buffer) == 0:
                        raise RuntimeError(
                            f"Buffer exhausted while packing batch {row_idx+1}/{self.batch_size}. "
                            f"Increase buffer_size (current: {self.buffer_size}) or prefetch_factor."
                        )

                    doc = self.doc_buffer.pop(
                        0
                    )  # shortest doc (optimization: min waste)
                    cropped = doc.tokens[:space]
                    try:
                        self.row_buffer[row_idx, pos : pos + space] = torch.tensor(
                            cropped, dtype=torch.long
                        )
                        batch_processed_tokens += space
                        batch_cropped_tokens += doc.length - space
                        pos += space
                    except Exception as e:
                        # handle corrupted token IDs gracefully
                        warnings.warn(
                            f"Invalid tokens while cropping document: {e}",
                            RuntimeWarning,
                        )
                        batch_corrupted_docs += 1
                        continue  # skip and try next doc

        # autoregressive split (nanochat-style): input[t] → target[t]
        inputs = self.row_buffer[:, :-1].clone()  # (B, T): tokens [0, T-1]
        targets = self.row_buffer[:, 1:].clone()  # (B, T): tokens [1, T]

        # Return per-batch stats (deltas only, not cumulative)
        batch_stats = {
            "total_tokens": batch_total_tokens,
            "cropped_tokens": batch_cropped_tokens,
            "processed_tokens": batch_processed_tokens,
            "dropped_tokens": batch_dropped_tokens,
            "dropped_files": batch_dropped_files,
            "buffer_overflows": batch_buffer_overflows,
            "corrupted_docs": batch_corrupted_docs,
            "empty_docs": batch_empty_docs,
        }

        return inputs, targets, batch_stats  # CPU tensors (workers can't access GPU)


class FinewebEduParquetBOSDataloader:
    """
    BOS-aligned dataloader with best-fit packing, built on PyTorch DataLoader.

    Features: parallel I/O (num_workers), prefetching, pin_memory, persistent workers.

    Usage:
        loader = FinewebEduParquetBOSDataloader(data_dir, batch_size=16, block_size=1024)
        for inputs, targets in loader:
            loss = model(inputs, targets)
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
        persistent_workers: bool = True,
        depth: Optional[int] = None,
    ):
        """
        Initialize BOS dataloader with best-fit packing.

        Args:
            data_dir: Parquet files directory
            batch_size: Number of sequences (B)
            block_size: Sequence length (T)
            ddp_world_size: Number of GPUs
            ddp_rank: Current GPU rank
            split: 'train' (all but last file) or 'val' (last file only)
            buffer_size: Document buffer for best-fit (~1000 works well)
            num_workers: Parallel I/O workers (4 is good default)
            persistent_workers: Keep workers alive across epochs (saves startup)
            depth: Model depth for auto-scaling dataloader_batch_size and prefetch_factor (None = use defaults)
        """
        # input validation
        if not os.path.isdir(data_dir):
            raise ValueError(f"data_dir must be a valid directory, got {data_dir}")
        if batch_size <= 0:
            raise ValueError(f"batch_size must be positive, got {batch_size}")
        if block_size <= 0:
            raise ValueError(f"block_size must be positive, got {block_size}")
        if ddp_world_size <= 0:
            raise ValueError(f"ddp_world_size must be positive, got {ddp_world_size}")
        if ddp_rank < 0 or ddp_rank >= ddp_world_size:
            raise ValueError(
                f"ddp_rank must be in [0, {ddp_world_size}), got {ddp_rank}"
            )
        if split not in ("train", "val"):
            raise ValueError(f"split must be 'train' or 'val', got {split}")
        if buffer_size <= 0:
            raise ValueError(f"buffer_size must be positive, got {buffer_size}")
        if num_workers < 0:
            raise ValueError(f"num_workers must be non-negative, got {num_workers}")

        # Convert torch.device to string if needed
        if isinstance(device, torch.device):
            device = str(device)

        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.split = split

        tokenizer, _ = get_custom_tokenizer()

        # validate tokenizer has BOS token
        try:
            bos_tokens = tokenizer.encode("<|bos|>", allowed_special="all")
            if not bos_tokens:
                raise ValueError("Tokenizer does not have a valid BOS token")
            bos_token_id = bos_tokens[0]
        except Exception as e:
            raise ValueError(f"Failed to get BOS token from tokenizer: {e}")

        # Determine prefetch_factor based on depth before printing
        # Scale inversely with model depth (Heuristic)
        # Deeper models → smaller batch → less data consumed → need fewer docs
        if depth is not None:
            # Inverse scaling with depth (matches model_setup.py batch_size scaling)
            # Conservative multipliers for FineWeb-Edu (long docs)
            # Format: max_depth: (multiplier, min_docs, prefetch_factor)
            depth_scaling_config = {
                8: (4, 256, 1),  # shallow models: high throughput, aggressive prefetch
                10: (4, 256, 1),  #
                14: (3, 128, 1),  # medium models: moderate doc buffer
                18: (2, 128, 1),  # deeper models: reduced buffer, conservative prefetch
                22: (2, 128, 1),  # deep models: minimal buffer
                float("inf"): (2, 32, 1),  # very deep models: minimal buffer
            }

            # Find appropriate config for this depth
            for max_depth, (multiplier, min_docs, pf) in depth_scaling_config.items():
                if depth <= max_depth:
                    dataloader_multiplier = multiplier
                    min_docs_threshold = min_docs
                    prefetch_factor = pf
                    break
        else:
            # Defaults for when depth is not specified
            dataloader_multiplier = 4
            min_docs_threshold = 64
            prefetch_factor = 2

        if master_process:
            print(f"\n{'='*80}")
            print(f"📚 BOS Dataloader ({split})")
            print(f"{'='*80}")
            print(f"Data: {data_dir}")
            print(
                f"Batch: {batch_size} × {block_size}, Buffer: {buffer_size}, BOS: {bos_token_id}"
            )
            print(f"Workers: {num_workers}, Persistent: {persistent_workers}")

        # get parquet files and split: train=all but last, val=last only
        parquet_files = list_parquet_files(data_dir)
        if not parquet_files:
            raise FileNotFoundError(f"No parquet files in {data_dir}")

        split_files = parquet_files[:-1] if split == "train" else parquet_files[-1:]
        if not split_files:
            raise ValueError(f"No files for split '{split}' (need ≥2 total files)")

        if master_process:
            print(
                f"Files: {len(parquet_files)} total, using {len(split_files)} for {split}"
            )
            print(f"{'='*80}\n")

        # create dataset: handles streaming, tokenization, and sharding
        dataset = ParquetDocumentDataset(
            split_files, tokenizer, ddp_rank, ddp_world_size, split
        )

        # create collator: packs variable-length docs → fixed (B, T) sequences
        self.collator = BestFitCollator(batch_size, block_size, buffer_size, device)

        # PyTorch edge case: num_workers=0 requires prefetch_factor=None, persistent_workers=False
        if num_workers == 0:
            prefetch_factor = None
            persistent_workers = False

        # DataLoader batch_size = documents per collate call (NOT model batch size!)
        # want enough docs for good best-fit, but not so many that buffer overflows
        # Use the dataloader_multiplier and min_docs_threshold determined above
        dataloader_batch_size = max(
            int(batch_size * dataloader_multiplier), min_docs_threshold
        )

        if master_process and depth is not None:
            print(
                f"   DataLoader batch size: {dataloader_batch_size} (depth={depth}, multiplier={dataloader_multiplier}×, min={min_docs_threshold})"
            )

        # pin_memory enables async CPU→GPU transfers (DMA), only works with workers>0
        use_pin_memory = device.startswith("cuda") and num_workers > 0

        # tie it all together: dataset streams docs, workers parallelize I/O,
        # collator packs into batches, prefetch hides latency
        self.dataloader = DataLoader(
            dataset,
            batch_size=dataloader_batch_size,
            collate_fn=self.collator,
            num_workers=num_workers,
            prefetch_factor=prefetch_factor,
            pin_memory=use_pin_memory,
            persistent_workers=persistent_workers,
            drop_last=False,
        )

        self._iterator = None

        if master_process:
            pin_status = "✓" if use_pin_memory else "✗"
            print(f"✓ DataLoader ready (pin_memory: {pin_status})\n")

        self.aggregated_stats = {
            "total_tokens": 0,
            "cropped_tokens": 0,
            "processed_tokens": 0,
            "dropped_tokens": 0,
            "dropped_files": 0,
            "buffer_overflows": 0,
            "corrupted_docs": 0,
            "empty_docs": 0,
        }

    def __iter__(self):
        self._iterator = iter(self.dataloader)
        return self

    def __next__(self):
        if self._iterator is None:
            self._iterator = iter(self.dataloader)
        inputs, targets, batch_stats = next(self._iterator)
        for key, value in batch_stats.items():
            self.aggregated_stats[key] += value
        return inputs, targets

    def reset(self):
        """
        Reset the dataloader iterator to start from the beginning.

        This is more efficient than creating a fresh dataloader for validation,
        as it preserves the worker processes and just resets the iteration state.
        """
        if self._iterator is not None:
            # Delete the old iterator to clean up resources
            del self._iterator
        # Create a new iterator (workers are persistent, so this is fast)
        self._iterator = iter(self.dataloader)

    def get_stats(self) -> dict:
        """Returns aggregated stats from all batches (across all workers)."""
        cropped_tokens_pct = (
            100.0
            * self.aggregated_stats["cropped_tokens"]
            / max(1, self.aggregated_stats["total_tokens"])
        )
        dropped_tokens_pct = (
            100.0
            * self.aggregated_stats["dropped_tokens"]
            / max(1, self.aggregated_stats["total_tokens"])
        )
        total_waste = (
            self.aggregated_stats["cropped_tokens"]
            + self.aggregated_stats["dropped_tokens"]
        )
        total_waste_pct = (
            100.0 * total_waste / max(1, self.aggregated_stats["total_tokens"])
        )

        return {
            "total_tokens": self.aggregated_stats["total_tokens"],
            "processed_tokens": self.aggregated_stats["processed_tokens"],
            "cropped_tokens": self.aggregated_stats["cropped_tokens"],
            "cropped_tokens_pct": cropped_tokens_pct,
            "crop_percentage": cropped_tokens_pct,  # backward compatibility alias
            "dropped_tokens": self.aggregated_stats["dropped_tokens"],
            "dropped_tokens_pct": dropped_tokens_pct,
            "total_waste_pct": total_waste_pct,
            "dropped_files": self.aggregated_stats["dropped_files"],
            "buffer_overflows": self.aggregated_stats["buffer_overflows"],
            "corrupted_docs": self.aggregated_stats["corrupted_docs"],
            "empty_docs": self.aggregated_stats["empty_docs"],
            "buffer_size": 0,  # Not tracked in aggregated stats (worker-local state)
        }

    def cleanup(self):
        """
        Gracefully cleanup dataloader resources (nanochat-style).

        Stops iterator, signals workers to finish, and releases resources.
        Safe to call multiple times (idempotent).
        """
        # Stop iterator to signal workers we're done
        if self._iterator is not None:
            try:
                # Delete iterator which signals DataLoader to stop prefetching
                del self._iterator
            except Exception:
                pass  # already cleaned up
            finally:
                self._iterator = None

        # Let DataLoader's internal cleanup handle worker shutdown
        # PyTorch DataLoader with persistent_workers=True will properly shutdown workers
        # when the iterator is deleted and garbage collected

    def __enter__(self):
        """Context manager support for automatic cleanup"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Cleanup on context exit"""
        self.cleanup()
        return False  # don't suppress exceptions
