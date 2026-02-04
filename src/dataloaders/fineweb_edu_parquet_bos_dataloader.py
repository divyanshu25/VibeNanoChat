"""
Parquet-based BOS-aligned dataloader (nanochat-style implementation).

This module provides a BOS-aligned data loader that reads from Parquet shards,
tokenizes documents on-the-fly, and packs them using best-fit cropping algorithm.

This is a direct port of nanochat's approach:
- Reads from Parquet files with 'text' column
- Each document is tokenized on-the-fly with BOS token prepended
- Best-fit algorithm minimizes token waste (~35% for longer documents, ~3% for shorter documents)
- 100% batch utilization (no padding)
- Supports DDP and multi-epoch training

Why BOS Alignment?
    Traditional dataloaders concatenate all documents and randomly slice them into sequences.
    This means sequences often start mid-document, making it harder for the model to learn
    document boundaries. BOS alignment ensures EVERY sequence starts at a document boundary
    with a BOS token, teaching the model that <|bos|> signifies a fresh start.

Best-Fit Packing Algorithm:
    Given a sequence of length T and a buffer of N documents:
    1. Greedily select the LARGEST document that fits in remaining space
    2. If no document fits entirely, crop the SHORTEST document to fill exactly
    3. Repeat until sequence is full (T+1 tokens)

    This minimizes waste compared to naive first-fit or random selection.
    Example with T=1024 and docs of length [100, 500, 800]:
        - Pack 800 → 224 remaining
        - Pack 100 → 124 remaining
        - Crop 500 to 124 → Full (only 376 tokens wasted = 27%)

Architecture:
    Parquet Files → Document Iterator → Tokenizer → Document Buffer → Best-Fit Packing → Batches

Example:
    >>> dataloader = FinewebEduParquetBOSDataloader(
    ...     data_dir="/sensei-fs/users/divgoyal/fineweb_edu_parquet",
    ...     batch_size=16,
    ...     block_size=1024,
    ...     split="train"
    ... )
    >>> for inputs, targets in dataloader:
    ...     # Every inputs[i, 0] is BOS token (50257)
    ...     loss = model(inputs, targets)
"""

import glob
import os

import pyarrow.parquet as pq
import torch

from gpt_2.utils import get_custom_tokenizer


def list_parquet_files(data_dir):
    """
    List all Parquet files in directory, sorted by name.

    Args:
        data_dir: Directory containing .parquet files (e.g., shard_00000.parquet, shard_00001.parquet, ...)

    Returns:
        list: Sorted list of full paths to .parquet files

    Note:
        - Files are sorted alphabetically, so shard_00000 comes before shard_00001
        - Skips .tmp files (used during dataset creation)
        - Nanochat convention: last file is validation, rest are training
    """
    parquet_files = sorted(
        [
            filepath
            for filepath in glob.glob(os.path.join(data_dir, "*.parquet"))
            if not filepath.endswith(".tmp")  # Skip temporary files
        ]
    )
    return parquet_files


def document_batches(
    split, parquet_paths, ddp_rank, ddp_world_size, tokenizer_batch_size=128
):
    """
    Infinite iterator over document batches from Parquet files.

    Yields batches of raw text documents, handles DDP sharding, and supports
    multi-epoch training by cycling through data indefinitely.

    Args:
        split: 'train' or 'val'
        parquet_paths: List of parquet file paths for this split
        ddp_rank: Rank of current process
        ddp_world_size: Total number of processes
        tokenizer_batch_size: Number of documents per batch

    Yields:
        tuple: (text_batch, epoch) where:
            - text_batch: List of document strings
            - epoch: Current epoch number (starts at 1)
    """
    assert split in ["train", "val"], "split must be 'train' or 'val'"

    epoch = 1  # Track current epoch for logging/curriculum learning

    # Infinite loop: allows multi-epoch training without recreating dataloader
    while True:
        # Iterate through all parquet shards (train: 180 shards, val: 1 shard)
        for shard_idx, filepath in enumerate(parquet_paths):
            parquet_file = pq.ParquetFile(
                filepath
            )  # this will read the parquet file and return a ParquetFile object

            # DDP Sharding Strategy: Each rank reads different row groups to avoid duplication
            # Example with 8 GPUs reading a file with 24 row groups:
            #   - Rank 0 reads: row groups 0, 8, 16
            #   - Rank 1 reads: row groups 1, 9, 17
            #   - Rank 2 reads: row groups 2, 10, 18
            #   - ... etc
            # This ensures each GPU sees different data with no overlap
            # Use range(start, stop, step) to iterate: start at rank, skip by world_size
            for row_group_idx in range(
                ddp_rank, parquet_file.num_row_groups, ddp_world_size
            ):
                # Read a single row group (contains ~1024 documents)
                row_group = parquet_file.read_row_group(row_group_idx)
                texts = row_group.column(
                    "text"
                ).to_pylist()  # Extract text column as Python list

                # Yield documents in smaller batches for efficient parallel tokenization
                # Example: If row group has 1024 docs and batch_size=128:
                #   - Yields 8 batches of 128 documents each
                for start_idx in range(0, len(texts), tokenizer_batch_size):
                    batch = texts[start_idx : start_idx + tokenizer_batch_size]
                    yield batch, epoch

        # After reading all shards once, increment epoch counter
        epoch += 1


class FinewebEduParquetBOSDataloader:
    """
    BOS-aligned dataloader with best-fit cropping (nanochat implementation).

    This loader reads Parquet shards, tokenizes documents on-the-fly, and packs
    them into batches using best-fit algorithm to minimize token waste.

    Key Features:
        1. BOS Alignment: Every sequence starts with a document boundary (BOS token)
        2. Best-Fit Packing: Minimizes token waste by intelligently selecting documents
        3. DDP Support: Each rank reads different row groups for parallel training
        4. Multi-Epoch: Infinite iterator automatically cycles through data
        5. Efficient Memory: Pre-allocated buffers and pinned memory for fast GPU transfer

    Algorithm Overview:
        Parquet Files → Row Groups (DDP sharded) → Documents → Tokenize with BOS →
        Document Buffer → Best-Fit Packing → Batches (B, T)

    Attributes:
        batch_size (int): Number of sequences per batch (B)
        block_size (int): Length of each sequence (T)
        device (str): Device ('cuda' or 'cpu')
        buffer_size (int): Number of documents to buffer for best-fit
        tokenizer: Custom tokenizer with special tokens
        doc_buffer (list): Buffer of tokenized documents
        batches (iterator): Document batch iterator
        current_epoch (int): Current epoch number
        stats_total_tokens (int): Total tokens processed
        stats_cropped_tokens (int): Tokens discarded due to cropping
    """

    def __init__(
        self,
        data_dir,
        batch_size,
        block_size,
        ddp_world_size=1,
        ddp_rank=0,
        split="train",
        master_process=True,
        buffer_size=1000,
        device="cuda",
        tokenizer_threads=4,
        tokenizer_batch_size=128,
    ):
        """
        Initialize the Parquet BOS-aligned dataloader.

        Args:
            data_dir (str): Directory containing shard_NNNNN.parquet files
            batch_size (int): Batch size (B)
            block_size (int): Sequence length (T)
            ddp_world_size (int): Number of processes
            ddp_rank (int): Current process rank
            split (str): 'train' or 'val'
            master_process (bool): Whether this is master process
            buffer_size (int): Document buffer size for best-fit
            device (str): 'cuda' or 'cpu'
            tokenizer_threads (int): Threads for parallel tokenization
            tokenizer_batch_size (int): Documents per tokenization batch
        """
        # Store configuration parameters (only those used after init)
        self.batch_size = batch_size
        self.block_size = block_size
        self.buffer_size = buffer_size  # Number of docs to keep in memory for best-fit
        self.device = device
        self.data_dir = data_dir  # Store for compute_val_steps
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.split = split
        self.tokenizer_batch_size = tokenizer_batch_size

        assert split in ["train", "val"], "split must be 'train' or 'val'"

        # Initialize tokenizer
        # BOS (Beginning of Sequence) token is prepended to each document
        self.tokenizer, _ = get_custom_tokenizer()
        bos_token_id = self.tokenizer.encode("<|bos|>", allowed_special="all")[0]

        if master_process:
            print(f"\n{'='*80}")
            print(f"📚 Parquet BOS-Aligned Dataloader ({split})")
            print(f"{'='*80}")
            print(f"Data directory: {data_dir}")
            print(f"Batch size: {batch_size}")
            print(f"Block size: {block_size}")
            print(f"Buffer size: {buffer_size}")
            print(f"BOS token ID: {bos_token_id}")
            print(f"Tokenizer threads: {tokenizer_threads}")

        # Verify data directory exists and has parquet files
        # Cache parquet files list to avoid repeated filesystem calls
        self.parquet_files = list_parquet_files(data_dir)
        if len(self.parquet_files) == 0:
            raise FileNotFoundError(f"No parquet files found in {data_dir}")

        # Split files: train gets all but last, val gets only last file
        if split == "train":
            self.split_parquet_files = self.parquet_files[:-1]
        else:
            self.split_parquet_files = self.parquet_files[-1:]

        if master_process:
            print(f"Found {len(self.parquet_files)} parquet files")
            if split == "train":
                print(
                    f"Using files: {os.path.basename(self.parquet_files[0])} to {os.path.basename(self.parquet_files[-2])}"
                )
            else:
                print(f"Using file: {os.path.basename(self.parquet_files[-1])}")
            print(f"{'='*80}\n")

        # Pre-allocate buffers for efficient batching (avoids repeated allocations)
        # Memory layout strategy:
        #   1. row_buffer: Working buffer on CPU for packing documents
        #   2. cpu_buffer: Pinned CPU memory for fast HtoD transfer
        #   3. gpu_buffer: Final GPU memory for training

        use_cuda = device == "cuda"

        # Row buffer: (batch_size, T+1) - used for assembling sequences with best-fit
        # We need T+1 tokens because:
        #   - T tokens for input (x)
        #   - T tokens for target (y), which is input shifted by 1
        # Shape example with B=16, T=1024: (16, 1025)
        self.row_buffer = torch.empty((batch_size, block_size + 1), dtype=torch.long)

        # CPU buffer: Flat buffer holding both inputs and targets (2 * B * T total)
        # We use pinned memory (page-locked) for faster GPU transfers
        # Shape: (2 * batch_size * block_size,) - will be split and reshaped
        self.cpu_buffer = torch.empty(
            2 * batch_size * block_size, dtype=torch.long, pin_memory=use_cuda
        )

        # GPU buffer: Same layout as CPU buffer, but on GPU device
        # This is the final destination for training data
        self.gpu_buffer = torch.empty(
            2 * batch_size * block_size, dtype=torch.long, device=device
        )

        # Create views into buffers (no data copying, just different shapes)
        # This allows us to:
        #   1. Fill buffers as flat arrays (efficient)
        #   2. Access them as 2D tensors (convenient)

        # CPU views: First half is inputs, second half is targets
        self.cpu_inputs = self.cpu_buffer[: batch_size * block_size].view(
            batch_size, block_size
        )  # Shape: (B, T)
        self.cpu_targets = self.cpu_buffer[batch_size * block_size :].view(
            batch_size, block_size
        )  # Shape: (B, T)

        # GPU views: Same split as CPU
        self.gpu_inputs = self.gpu_buffer[: batch_size * block_size].view(
            batch_size, block_size
        )  # Shape: (B, T)
        self.gpu_targets = self.gpu_buffer[batch_size * block_size :].view(
            batch_size, block_size
        )  # Shape: (B, T)

        # Initialize document buffer and iterator
        # doc_buffer: List of tokenized documents waiting to be packed
        #   Example: [[50257, 464, 2068, ...], [50257, 1135, 318, ...], ...]
        #   Each inner list is a tokenized document starting with BOS
        self.doc_buffer = []
        self.current_epoch = 1

        # Create infinite iterator that yields document batches
        # This handles reading parquet files, DDP sharding, and multi-epoch cycling
        self.batches = document_batches(
            split,
            self.split_parquet_files,
            ddp_rank,
            ddp_world_size,
            tokenizer_batch_size,
        )

        # Row Group (1024 documents)
        #     ↓
        # Split into chunks of tokenizer_batch_size=128
        #     ↓
        # Yields 8 batches: [128 docs, 128 docs, 128 docs, ..., 128 docs]
        #     ↓
        # Each batch gets tokenized
        #     ↓
        # Documents go into doc_buffer
        #     ↓
        # Best-fit packing creates training batches of size batch_size=16

        # Statistics for tracking packing efficiency
        # Goal: Keep crop_percentage around 35% (mentioned in nanochat docs)
        self.stats_total_tokens = 0  # Total tokens used (including cropped)
        self.stats_cropped_tokens = 0  # Tokens discarded due to cropping

        # Pre-fill buffer before starting training
        # This ensures the best-fit algorithm has enough choices from the start
        if master_process:
            print("Pre-filling document buffer...")
        self._refill_buffer(initial=True)
        if master_process:
            print(f"✓ Buffer initialized with {len(self.doc_buffer)} documents\n")

    def _refill_buffer(self, initial=False):
        """
        Refill the document buffer by fetching and tokenizing new documents.

        Args:
            initial (bool): If True, fills to buffer_size. Otherwise adds half.
        """
        # Buffer management strategy:
        #   - Initial fill: Fill completely to buffer_size (e.g., 1000 docs)
        #   - Subsequent fills: Only add buffer_size // 2 (e.g., 500 docs)
        # This keeps a mix of old and new documents for better best-fit choices
        target_size = self.buffer_size if initial else self.buffer_size // 2
        docs_to_add = target_size - len(self.doc_buffer)

        # Buffer is still full enough, don't need to refill yet
        if docs_to_add <= 0:
            return

        documents_added = 0
        while documents_added < docs_to_add:
            # Fetch next batch of raw text documents from parquet iterator
            # text_batch: list of strings (raw documents)
            # epoch: current epoch number (for tracking/logging)
            text_batch, epoch = next(self.batches)
            self.current_epoch = epoch

            # Tokenize all documents in the batch
            # Why prepend BOS? Ensures model learns that BOS starts a new document
            # This is critical for generation and helps with document boundaries
            tokenized_documents = []
            for document_text in text_batch:
                # Convert text to tokens: "<|bos|>" + document_text
                # Example: "<|bos|>The quick brown fox..." -> [50257, 464, 2068, ...]
                token_ids = self.tokenizer.encode(
                    f"<|bos|>{document_text}", allowed_special="all"
                )
                tokenized_documents.append(token_ids)

            # Add tokenized documents to buffer
            # Buffer is a list of token lists: [[tok1, tok2, ...], [tok1, tok2, ...], ...]
            for token_ids in tokenized_documents:
                if (
                    len(token_ids) > 0
                ):  # Skip empty documents (shouldn't happen but safe)
                    self.doc_buffer.append(token_ids)
                    documents_added += 1
                    if documents_added >= docs_to_add:
                        break

    def next_batch(self):
        """
        Get the next BOS-aligned batch using best-fit cropping.

        Returns:
            tuple: (inputs, targets) where:
                - inputs: Tensor of shape (batch_size, block_size)
                - targets: Tensor of shape (batch_size, block_size)

        Each row starts with BOS token at a document boundary.
        """
        # Fill each row in the batch using best-fit packing algorithm
        # Goal: Pack documents into fixed-length sequences with minimal waste
        for row_idx in range(self.batch_size):
            current_position = 0  # Current position in the row (0 to block_size + 1)

            # Keep packing documents until row is full
            while current_position < self.block_size + 1:
                # Refill buffer if it's getting low (below 25% capacity)
                # This ensures we always have enough documents for good best-fit choices
                if len(self.doc_buffer) < self.buffer_size // 4:
                    self._refill_buffer()

                remaining_tokens = (
                    self.block_size + 1 - current_position
                )  # How many tokens can still fit

                # BEST-FIT ALGORITHM: Find the LARGEST document that fits completely
                # Example: If remaining=500, and we have docs of length [100, 450, 600, 200]:
                #   - Doc 100: fits, but not largest
                #   - Doc 450: fits, and is largest so far ✓
                #   - Doc 600: doesn't fit (too big)
                #   - Doc 200: fits, but smaller than 450
                # Best choice: 450 (minimizes leftover space = 50 tokens)
                best_fit_idx = -1
                best_fit_length = 0
                for buffer_idx, token_ids in enumerate(self.doc_buffer):
                    document_length = len(token_ids)
                    # Document fits AND is larger than current best
                    if (
                        document_length <= remaining_tokens
                        and document_length > best_fit_length
                    ):
                        best_fit_idx = buffer_idx
                        best_fit_length = document_length

                if best_fit_idx >= 0:
                    # SUCCESS: Found a document that fits entirely (no cropping needed)
                    token_ids = self.doc_buffer.pop(best_fit_idx)  # Remove from buffer
                    document_length = len(token_ids)
                    # Copy tokens into row buffer at current position
                    self.row_buffer[
                        row_idx, current_position : current_position + document_length
                    ] = torch.tensor(token_ids, dtype=torch.long)
                    current_position += document_length  # Advance position
                    self.stats_total_tokens += document_length  # Track for statistics
                else:
                    # FALLBACK: No document fits entirely, must crop one
                    # Strategy: Crop the SHORTEST document to minimize waste
                    # Why shortest? Large docs might fit in future rows better

                    if len(self.doc_buffer) == 0:
                        # Emergency: buffer is empty, refill immediately
                        self._refill_buffer()
                        continue

                    # Find shortest document in buffer
                    shortest_doc_idx = min(
                        range(len(self.doc_buffer)),
                        key=lambda idx: len(self.doc_buffer[idx]),
                    )
                    token_ids = self.doc_buffer.pop(shortest_doc_idx)

                    # Crop document to fit exactly in remaining space
                    # Example: If remaining=50 and doc has 200 tokens:
                    #   - Use first 50 tokens: doc[:50]
                    #   - Discard last 150 tokens (waste)
                    cropped_tokens = token_ids[:remaining_tokens]
                    self.row_buffer[
                        row_idx, current_position : current_position + remaining_tokens
                    ] = torch.tensor(cropped_tokens, dtype=torch.long)
                    self.stats_total_tokens += remaining_tokens
                    self.stats_cropped_tokens += (
                        len(token_ids) - remaining_tokens
                    )  # Track waste
                    current_position += remaining_tokens  # Row is now full

        # Now row_buffer contains (B, T+1) tokens for each sequence
        # Split into inputs and targets using sequence shift:
        #   - inputs: first T tokens  [0:T]   (what model sees)
        #   - targets: last T tokens [1:T+1]  (what model should predict)
        # This creates autoregressive training pairs
        self.cpu_inputs.copy_(self.row_buffer[:, :-1])  # (B, T) - remove last token
        self.cpu_targets.copy_(self.row_buffer[:, 1:])  # (B, T) - remove first token

        # Transfer from CPU to GPU in one efficient batch
        # non_blocking=True allows CPU to continue while GPU copies (pipelining)
        use_cuda = self.device == "cuda"
        self.gpu_buffer.copy_(self.cpu_buffer, non_blocking=use_cuda)

        # Return GPU tensors (views into gpu_buffer)
        return self.gpu_inputs, self.gpu_targets

    def __iter__(self):
        """
        Make dataloader iterable.

        This allows using the dataloader in a for loop:
            for inputs, targets in dataloader:
                loss = model(inputs, targets)
        """
        return self

    def __next__(self):
        """
        Get next batch (for iterator protocol).

        Called automatically when iterating:
            for inputs, targets in dataloader:  # Calls __next__ each iteration
                ...

        Returns:
            tuple: (inputs, targets) both of shape (batch_size, block_size)
        """
        return self.next_batch()

    def get_stats(self):
        """
        Get token usage statistics for monitoring packing efficiency.

        Returns:
            dict: Statistics with:
                - total_tokens: Total tokens processed
                - cropped_tokens: Tokens discarded due to cropping
                - crop_percentage: Percentage of tokens wasted (target: ~35%)
                - current_epoch: Current epoch number

        Interpretation:
            - Lower crop_percentage = better packing efficiency
            - Nanochat reports ~35% cropping for longer documents
            - Very low (<5%) suggests most docs are short
            - Very high (>50%) suggests poor buffer size or doc length distribution
        """
        crop_pct = 100.0 * self.stats_cropped_tokens / max(1, self.stats_total_tokens)
        return {
            "total_tokens": self.stats_total_tokens,
            "cropped_tokens": self.stats_cropped_tokens,
            "crop_percentage": crop_pct,
            "current_epoch": self.current_epoch,
        }
