"""
OpenWebText DataLoader for GPT-2 Training

This module provides a data loader for the OpenWebText dataset, designed for training
GPT-2 models. The loader handles memory-mapped binary files containing tokenized data
and supports distributed training across multiple processes.

The OpenWebText dataset contains ~9B training tokens and ~4M validation tokens,
stored as continuous binary files (train.bin and val.bin) with uint16 dtype.

Example:
    >>> dataloader = OpenWebtextDataloader(
    ...     data_dir="/sensei-fs/users/divgoyal/openwebtext",
    ...     batch_size=128,
    ...     block_size=1024,
    ...     split="train"
    ... )
    >>> x, y = dataloader.next_batch()
    >>> print(f"Input shape: {x.shape}, Target shape: {y.shape}")
    Input shape: torch.Size([128, 1024]), Target shape: torch.Size([128, 1024])
"""

import os

import numpy as np
import torch


class OpenWebtextDataloader:
    """
    A data loader for the OpenWebText dataset optimized for GPT-2 training.

    This class manages loading and batching of tokenized data from binary files.
    It supports distributed training by handling different data partitions for
    different processes and provides efficient sequential access to training data.

    The loader uses memory-mapped files for efficient access without loading
    the entire dataset into RAM.

    Attributes:
        data_dir (str): Directory containing train.bin and val.bin
        batch_size (int): Number of sequences per batch
        block_size (int): Length of each training sequence
        ddp_world_size (int): Total number of processes in distributed training
        ddp_rank (int): Rank of current process in distributed training
        split (str): Dataset split ('train' or 'val')
        tokens (np.memmap): Memory-mapped token data
        current_position (int): Current position within the token array
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
    ):
        """
        Initialize the OpenWebText data loader.

        Args:
            data_dir (str): Directory containing train.bin and val.bin files.
            batch_size (int): Number of sequences per batch.
            block_size (int): Length of each training sequence.
            ddp_world_size (int, optional): Total number of processes in
                distributed training. Defaults to 1.
            ddp_rank (int, optional): Rank of current process in distributed
                training. Defaults to 0.
            split (str, optional): Dataset split to use. Must be either
                'train' or 'val'. Defaults to "train".
            master_process (bool, optional): Whether this is the master process.
                Controls logging output. Defaults to True.

        Raises:
            AssertionError: If split is not 'train' or 'val', or if the
                binary file is not found in the data directory.
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.block_size = block_size
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.split = split
        assert self.split in ["train", "val"], "split must be either train or val"

        # Load the binary file using memory mapping
        data_file = os.path.join(data_dir, f"{split}.bin")
        assert os.path.exists(data_file), f"Data file not found: {data_file}"

        if master_process:
            print(f"Loading {split} data from {data_file}")

        # Memory-map the binary file (efficient, doesn't load all into RAM)
        try:
            self.tokens = np.memmap(data_file, dtype=np.uint16, mode="r")
        except OSError as e:
            # Fallback: Some filesystems (like network FS) may not support mmap
            if master_process:
                print(f"Warning: mmap failed ({e}), loading entire file into memory...")
            with open(data_file, "rb") as f:
                data = np.fromfile(f, dtype=np.uint16)
            self.tokens = data

        if master_process:
            num_tokens = len(self.tokens)
            size_mb = num_tokens * 2 / (1024 * 1024)  # uint16 = 2 bytes
            print(f"Loaded {num_tokens:,} tokens ({size_mb:.2f} MB) for {split} split")

        self.reset()

    def reset(self):
        """
        Reset the data loader to a random starting position.
        """
        self.current_position = self.batch_size * self.block_size * self.ddp_rank

    def next_batch(self):
        """
        Get the next batch of training data.

        Returns a batch of input-target pairs for training. The inputs are
        sequences of tokens, and the targets are the same sequences shifted
        by one position (for next-token prediction).

        Returns:
            tuple: A pair of tensors (x, y) where:
                - x (torch.Tensor): Input sequences with shape
                  (batch_size, block_size)
                - y (torch.Tensor): Target sequences with shape
                  (batch_size, block_size)

        Note:
            - The method automatically wraps around to the beginning when
              the end of the dataset is reached
            - In distributed training, each process gets a different subset
              of the data based on its rank
            - The position advances by batch_size * block_size * ddp_world_size
              to ensure no overlap between processes
        """
        # Extract batch from current position
        # Need block_size + 1 tokens to create input and target sequences
        buf = self.tokens[
            self.current_position : self.current_position
            + self.batch_size * self.block_size
            + 1
        ]

        # Convert to torch tensor
        buf = torch.tensor(buf.astype(np.int32), dtype=torch.long)

        # Create input (x) and target (y) sequences
        # x: tokens[0:block_size], y: tokens[1:block_size+1]
        x = buf[:-1].view(self.batch_size, self.block_size)  # inputs
        y = buf[1:].view(self.batch_size, self.block_size)  # targets (shifted by 1)

        # Advance position for next batch
        # Multiply by world_size to ensure different processes don't overlap
        self.current_position += self.batch_size * self.block_size * self.ddp_world_size

        # Wrap around to beginning if we reach the end
        if self.current_position + (
            self.batch_size * self.block_size * self.ddp_world_size + 1
        ) > len(self.tokens):
            self.current_position = self.batch_size * self.block_size * self.ddp_rank

        return x, y
