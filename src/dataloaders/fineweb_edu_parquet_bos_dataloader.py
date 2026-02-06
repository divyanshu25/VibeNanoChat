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
from typing import List, Optional, Tuple

import pyarrow.parquet as pq
import torch
from sortedcontainers import SortedList
from torch.utils.data import DataLoader, IterableDataset

from gpt_2.utils import get_custom_tokenizer


@functools.total_ordering  # auto-generates <=, >, >= from __eq__ and __lt__
class SortedDocument:
    """
    Document wrapper for length-based sorting in O(log n) binary search.

    This is the key to efficient best-fit packing. By wrapping documents in this class,
    we can maintain a sorted buffer (via SortedList) and use binary search to find
    the largest document that fits in remaining space - O(log n) instead of O(n).
    """

    __slots__ = (
        "tokens",
        "length",
    )  # memory optimization: prevents __dict__ allocation
    # saves ~56 bytes per doc (important with 1000s in buffer)

    def __init__(self, tokens: List[int]):
        self.tokens = tokens
        self.length = len(
            tokens
        )  # cache length for O(1) comparisons (called frequently)

    def __lt__(self, other):
        # comparison operator for sorting by length
        # supports both SortedDocument and int comparisons, which lets us do:
        # buffer.bisect_right(150)  <- find where a doc of length 150 would go
        other_len = other.length if isinstance(other, SortedDocument) else other
        return self.length < other_len

    def __eq__(self, other):
        # equality by length (not content) since we only care about packing efficiency
        other_len = other.length if isinstance(other, SortedDocument) else other
        return self.length == other_len

    def __len__(self):
        # allows len(doc) to work, which is more pythonic than doc.length
        return self.length


def list_parquet_files(data_dir: str) -> List[str]:
    """List all .parquet files, sorted by name (excludes .tmp files)."""
    pattern = os.path.join(data_dir, "*.parquet")
    # exclude .tmp files which might be in-progress writes from data preprocessing
    # sorted order ensures deterministic file iteration across runs
    return sorted(f for f in glob.glob(pattern) if not f.endswith(".tmp"))


class PackedParquetDataset(IterableDataset):
    """
    Parquet dataset with integrated best-fit packing.

    Yields packed batches directly instead of individual documents, giving better
    control over buffer management and consumption rate.

    Each yielded batch is (B, T+1) ready for autoregressive split.
    """

    def __init__(
        self,
        parquet_paths: List[str],
        tokenizer,
        batch_size: int,
        block_size: int,
        buffer_size: int,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        split: str = "train",
    ):
        super().__init__()
        self.parquet_paths = parquet_paths
        self.tokenizer = tokenizer
        self.batch_size = batch_size
        self.block_size = block_size
        self.buffer_size = buffer_size
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.split = split

    def _tokenize_document(self, text: str) -> Optional[List[int]]:
        """Tokenize a single document with BOS prefix and validation."""
        # skip empty or malformed text early - saves tokenization cost
        if not text or not isinstance(text, str):
            return None

        try:
            # prepend BOS token to every document - this is critical for the model to learn
            # document boundaries. without this, the model can't distinguish between
            # "end of doc A" and "start of doc B" when they're packed together
            tokens = self.tokenizer.encode(f"<|bos|>{text}", allowed_special="all")

            # empty tokenization can happen with special chars or whitespace-only docs
            if len(tokens) == 0:
                return None

            # sanity check: skip pathologically long documents (> 1M tokens ~= 750k words)
            # these are usually data errors (concatenated docs, corrupt parquet, etc)
            # and would dominate the buffer, causing poor packing efficiency
            if len(tokens) > 1000000:
                warnings.warn(
                    f"Skipping document with {len(tokens)} tokens (> 1M)",
                    RuntimeWarning,
                )
                return None

            return tokens
        except Exception as e:
            # tokenization can fail on malformed unicode, null bytes, etc
            # just skip these docs and continue - data robustness over strictness
            warnings.warn(f"Tokenization error: {e}", RuntimeWarning)
            return None

    def _read_row_group_texts(
        self, parquet_file, row_group_idx: int, filepath: str
    ) -> List[str]:
        """Read texts from a single row group with error handling."""
        try:
            # parquet files are organized into row groups (~1000-10000 rows each)
            # this allows parallel reading across workers without loading entire file
            row_group = parquet_file.read_row_group(row_group_idx)
            # extract just the "text" column and convert to python list
            # much faster than iterating rows, since parquet is columnar
            return row_group.column("text").to_pylist()
        except Exception as e:
            # corrupted row groups can happen from incomplete writes, disk errors, etc
            # skip the bad row group and continue - we have billions of docs anyway
            warnings.warn(
                f"Skipping corrupted row group {row_group_idx} in {filepath}: {e}",
                RuntimeWarning,
            )
            return []

    def _document_stream(
        self, shard_id: int, global_num_workers: int, rng: random.Random
    ):
        """Stream tokenized documents from parquet files with worker-based sharding."""
        # infinite epoch loop - we never run out of data, just cycle through files forever
        # this is the standard approach for large-scale pretraining (GPT-3 did this too)
        while True:
            # shuffle file order each epoch - prevents overfitting to file ordering
            # each worker uses its own seeded RNG (via shard_id), so shuffles are
            # deterministic but different across workers (good for diversity)
            shuffled_paths = self.parquet_paths.copy()
            rng.shuffle(shuffled_paths)

            for filepath in shuffled_paths:
                try:
                    # use context manager to ensure file is closed even if we break/continue
                    # parquet files keep OS file handles open, so this prevents fd leaks
                    with pq.ParquetFile(filepath) as parquet_file:
                        # SHARDING STRATEGY: strided access across row groups
                        # if we have 8 workers and 100 row groups, worker 0 reads 0,8,16,...
                        # worker 1 reads 1,9,17,... etc. This ensures:
                        # 1) no data duplication (each row group read by exactly one worker)
                        # 2) perfect load balancing (row groups evenly distributed)
                        # 3) no coordination needed between workers (stateless sharding)
                        for row_group_idx in range(
                            shard_id, parquet_file.num_row_groups, global_num_workers
                        ):
                            texts = self._read_row_group_texts(
                                parquet_file, row_group_idx, filepath
                            )

                            # tokenize each text document and yield valid tokens
                            # this is a generator, so tokenization happens lazily on-demand
                            # (only tokenize when buffer needs more docs)
                            for text in texts:
                                tokens = self._tokenize_document(text)
                                if tokens is not None:
                                    yield tokens
                except Exception as e:
                    # file-level errors (permission denied, file deleted, etc)
                    # just skip the whole file and move on - robustness over strictness
                    warnings.warn(f"Error reading {filepath}: {e}", RuntimeWarning)
                    continue

    def _fill_buffer(
        self,
        doc_buffer,
        doc_stream,
        target_docs: int,
        max_buffer: int,
        batch_stats: dict,
    ):
        """Fill document buffer to target size, respecting max limit."""
        # SMART BUFFER MANAGEMENT: only fill when needed, stop at max capacity
        # buffer naturally shrinks through consumption during packing

        # don't fill if we're already above target (buffer has enough docs)
        if len(doc_buffer) >= target_docs:
            return

        # fill to target, but don't exceed max_buffer (leave some headroom)
        while len(doc_buffer) < target_docs:
            # safety check: stop if we're approaching max capacity
            # leave 10% headroom to avoid edge cases where we overshoot slightly
            if len(doc_buffer) >= max_buffer * 0.9:
                break

            try:
                # pull next tokenized document from stream (lazy, on-demand)
                tokens = next(doc_stream)
                # wrap in SortedDocument for O(log n) length-based lookups
                # add() maintains sorted order via binary search insertion
                doc_buffer.add(SortedDocument(tokens))
                # track total tokens we've seen (for computing crop percentage later)
                batch_stats["total_tokens"] += len(tokens)
            except StopIteration:
                # document stream is exhausted (shouldn't happen with infinite loop,
                # but handle gracefully just in case)
                break

    def _pack_row(
        self,
        doc_buffer,
        row_buffer,
        row_idx: int,
        doc_stream,
        min_docs_for_batch: int,
        max_buffer: int,
        batch_stats: dict,
    ):
        """Pack a single row using best-fit algorithm."""
        pos = 0  # current write position in the row
        target_length = (
            self.block_size + 1
        )  # +1 because we need input[:-1] and target[1:]

        # CORE PACKING ALGORITHM: fill each row to exactly target_length with no padding
        # this gives 100% token utilization (every token is trained on)
        while pos < target_length:
            space = target_length - pos  # how many tokens we still need

            # BEST-FIT SEARCH: find the largest document that fits in remaining space
            # bisect_right(space) finds insertion point for a doc of length 'space'
            # subtracting 1 gives us the largest doc with length <= space
            # this is O(log n) binary search on the sorted buffer (very fast!)
            # example: space=100, buffer has docs of length [5, 20, 50, 80, 150, 200]
            #          bisect_right(100) = 4, so best_idx = 3, doc_length = 80 (perfect!)
            best_idx = doc_buffer.bisect_right(space) - 1

            if 0 <= best_idx < len(doc_buffer):
                # CASE 1: found a document that fits entirely
                # pack it without cropping (preserves full document context)
                doc = doc_buffer.pop(best_idx)
                row_buffer[row_idx, pos : pos + doc.length] = torch.tensor(
                    doc.tokens, dtype=torch.long
                )
                pos += doc.length
                batch_stats["processed_tokens"] += doc.length
            else:
                # CASE 2: no document fits (all docs are longer than remaining space)
                # we must crop a document to fill the row exactly (100% utilization)
                if len(doc_buffer) == 0:
                    # buffer is empty - need to refill before we can crop
                    self._fill_buffer(
                        doc_buffer,
                        doc_stream,
                        min_docs_for_batch,
                        max_buffer,
                        batch_stats,
                    )
                    if len(doc_buffer) == 0:
                        raise RuntimeError("Buffer exhausted. Increase buffer_size.")
                    continue  # try packing again now that buffer is refilled

                # crop the SHORTEST document (minimize waste from cropping)
                # shortest doc is at index 0 since buffer is sorted by length
                doc = doc_buffer.pop(0)
                row_buffer[row_idx, pos : pos + space] = torch.tensor(
                    doc.tokens[:space], dtype=torch.long
                )
                batch_stats["processed_tokens"] += space  # tokens we actually use
                batch_stats["cropped_tokens"] += doc.length - space  # tokens we discard
                pos += space  # row is now full (pos == target_length)

    def __iter__(self):
        """Yield packed batches of shape (B, T+1)"""

        # ========== SETUP DISTRIBUTED SHARDING ==========
        # PyTorch DataLoader provides worker info for multi-process data loading
        # each worker reads a disjoint subset of data (no duplication)
        worker_info = torch.utils.data.get_worker_info()
        worker_id = 0 if worker_info is None else worker_info.id
        num_workers = 1 if worker_info is None else worker_info.num_workers

        # combine DDP sharding (across GPUs) with DataLoader sharding (across workers)
        # if we have 4 GPUs with 2 workers each, that's 8 total workers
        # each worker gets a unique shard_id in [0, 7]
        global_num_workers = self.ddp_world_size * num_workers
        shard_id = self.ddp_rank * num_workers + worker_id

        # log sharding setup once (only from the master worker to avoid spam)
        if self.ddp_rank == 0 and worker_id == 0:
            print(
                f"Sharding files across: {global_num_workers} total workers "
                f"({self.ddp_world_size} GPUs × {num_workers} workers/GPU)"
            )

        # ========== INITIALIZE PACKING STATE ==========
        # doc_buffer: sorted collection of documents for O(log n) best-fit search
        # we maintain docs sorted by length so bisect_right(space) finds best fit fast
        doc_buffer = SortedList()

        # row_buffer: pre-allocate a (B, T+1) tensor that we'll reuse each batch
        # shape is T+1 because autoregressive training needs input[:-1] and target[1:]
        row_buffer = torch.empty(
            (self.batch_size, self.block_size + 1), dtype=torch.long
        )

        # batch_stats: track packing efficiency and data quality per batch
        # these accumulate in the main dataloader class for end-of-training reporting
        batch_stats = {
            "total_tokens": 0,  # all tokens we've read from parquet files
            "cropped_tokens": 0,  # tokens discarded due to packing (unavoidable waste)
            "processed_tokens": 0,  # tokens actually trained on (what we want to maximize)
            "corrupted_docs": 0,  # docs with invalid tokens/encoding
            "empty_docs": 0,  # docs that tokenized to empty
        }

        # ========== CREATE INFINITE DOCUMENT STREAM ==========
        # each worker uses a seeded RNG (based on shard_id) for deterministic shuffling
        # this ensures reproducible training runs while maintaining per-worker diversity
        rng = random.Random(shard_id)
        doc_stream = self._document_stream(shard_id, global_num_workers, rng)

        # ========== MAIN PACKING LOOP ==========
        # SMART BUFFER SIZING STRATEGY:
        # - target_docs: optimal size for good packing efficiency (more choices for best-fit)
        # - max_buffer: hard safety limit to prevent OOM (100x buffer_size)
        # - _fill_buffer stops at 90% of max_buffer to stay under memory limits
        # - buffer naturally shrinks through consumption, then refills when below target
        #
        # Example: if buffer_size=1000, target=max(32, batch_size), max=100,000
        #   → we fill to ~32 docs initially
        #   → as we pack, buffer drains (docs consumed)
        #   → when buffer < 32, we refill to 32 (but never exceed 90k docs)
        target_docs = max(self.batch_size, 32)  # at least 32 docs for good packing
        max_buffer = self.buffer_size * 100  # 100x buffer_size as hard limit

        while True:  # infinite batches for infinite training
            # reset stats for this batch (we report per-batch deltas)
            for key in batch_stats:
                batch_stats[key] = 0

            # PROACTIVE BUFFER MANAGEMENT: fill to target, but stop at 90% of max
            # buffer naturally shrinks through consumption during packing, then we refill
            self._fill_buffer(
                doc_buffer, doc_stream, target_docs, max_buffer, batch_stats
            )

            # pack B rows using best-fit algorithm
            # each row is filled to exactly (block_size + 1) tokens with no padding
            # documents are packed greedily: largest-fit-first, then crop when nothing fits
            for row_idx in range(self.batch_size):
                self._pack_row(
                    doc_buffer,
                    row_buffer,
                    row_idx,
                    doc_stream,
                    target_docs,
                    max_buffer,
                    batch_stats,
                )

            # yield (B, T+1) batch and stats
            # clone() is necessary because row_buffer is reused across batches
            # without clone(), all batches would share the same underlying memory
            yield row_buffer.clone(), batch_stats.copy()


class SimpleCollator:
    """
    Simple collator for pre-packed batches.

    Just performs the autoregressive split and passes through stats.
    Packing is now handled in the dataset itself, so this is very lightweight.
    """

    def __call__(
        self, batch: List[Tuple[torch.Tensor, dict]]
    ) -> Tuple[torch.Tensor, torch.Tensor, dict]:
        """
        Perform autoregressive split on pre-packed batch.

        This is the standard language modeling setup: predict next token given previous tokens.
        We receive a batch of shape (B, T+1) and split it into inputs and targets.

        Args:
            batch: List of (packed_batch, stats) tuples (typically length 1)

        Returns:
            (inputs, targets, stats) where inputs[:, t] predicts targets[:, t]
        """
        # unpack - should be single item since dataset yields complete batches
        # (DataLoader batch_size=1 in this design, real batching happens in dataset)
        packed_batch, stats = batch[0]

        # AUTOREGRESSIVE SPLIT: create (input, target) pairs for next-token prediction
        # packed_batch is shape (B, T+1), we split it into:
        #   inputs:  tokens [0, 1, 2, ..., T-1]  <- what the model sees
        #   targets: tokens [1, 2, 3, ..., T]    <- what the model should predict
        # so inputs[:, t] is used to predict targets[:, t]
        #
        # example with T=4:
        #   packed_batch = [[5, 10, 15, 20, 25]]  (shape 1, 5)
        #   inputs  = [[5, 10, 15, 20]]           (shape 1, 4)
        #   targets = [[10, 15, 20, 25]]          (shape 1, 4)
        #   model predicts: 5→10, 10→15, 15→20, 20→25
        inputs = packed_batch[:, :-1]  # (B, T): tokens [0, T-1]
        targets = packed_batch[:, 1:]  # (B, T): tokens [1, T]

        return inputs, targets, stats


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
        persistent_workers: bool = True,
        num_workers: int = 4,
        prefetch_factor: int = 2,
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
            buffer_size: Document buffer for best-fit packing (larger = better packing efficiency)
            persistent_workers: Keep workers alive across epochs (saves startup)
            num_workers: Number of data loading workers (default: 4)
            prefetch_factor: Number of batches to prefetch per worker (default: 2)
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

        # Convert torch.device to string if needed
        if isinstance(device, torch.device):
            device = str(device)

        self.batch_size = batch_size
        self.block_size = block_size
        self.device = device
        self.split = split
        self.prefetch_factor = prefetch_factor

        tokenizer, _ = get_custom_tokenizer()

        # validate tokenizer has BOS token
        try:
            bos_tokens = tokenizer.encode("<|bos|>", allowed_special="all")
            if not bos_tokens:
                raise ValueError("Tokenizer does not have a valid BOS token")
            bos_token_id = bos_tokens[0]
        except Exception as e:
            raise ValueError(f"Failed to get BOS token from tokenizer: {e}")

        # Use fixed sensible defaults - packing now happens in dataset
        # which gives us better control over buffer management
        # num_workers and prefetch_factor are now arguments

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

        # Create dataset: handles streaming, tokenization, packing, and sharding
        # Packing is now integrated into the dataset for better control
        dataset = PackedParquetDataset(
            parquet_paths=split_files,
            tokenizer=tokenizer,
            batch_size=batch_size,
            block_size=block_size,
            buffer_size=buffer_size,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            split=split,
        )

        # Create collator: simple pass-through that does autoregressive split
        self.collator = SimpleCollator()

        # PyTorch edge case: num_workers=0 requires prefetch_factor=None, persistent_workers=False
        if num_workers == 0:
            prefetch_factor = None
            persistent_workers = False
        else:
            prefetch_factor = self.prefetch_factor

        # pin_memory enables async CPU→GPU transfers (DMA), only works with workers>0
        use_pin_memory = device.startswith("cuda") and num_workers > 0

        # DataLoader with batch_size=1 since batching happens in the dataset
        # Dataset yields complete packed batches, DataLoader just wraps them
        self.dataloader = DataLoader(
            dataset,
            batch_size=1,  # Dataset already yields packed batches
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

        return {
            "total_tokens": self.aggregated_stats["total_tokens"],
            "processed_tokens": self.aggregated_stats["processed_tokens"],
            "cropped_tokens": self.aggregated_stats["cropped_tokens"],
            "cropped_tokens_pct": cropped_tokens_pct,
            "crop_percentage": cropped_tokens_pct,  # backward compatibility alias
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
