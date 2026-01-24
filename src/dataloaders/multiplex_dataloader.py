"""
Multiplex DataLoader for Multiple Task Datasets

This module provides a PyTorch-based data loader that can multiplex multiple
existing dataloaders (e.g., ARC, GSM8K, MMLU) into a single DataLoader with
support for all PyTorch features like pin_memory, prefetch_factor, num_workers, etc.

The multiplexing strategy allows you to train on a mixture of datasets with
configurable sampling strategies (uniform, weighted, or proportional to dataset size).

Example:
    >>> from dataloaders.arc_dataloader import ARCDataLoader
    >>> from dataloaders.gsm8k_dataloader import GSM8KDataLoader
    >>> from dataloaders.mmlu_dataloader import MMLUDataLoader
    >>>
    >>> # Create individual dataset loaders
    >>> arc_loader = ARCDataLoader(subset="ARC-Easy", split="train")
    >>> gsm8k_loader = GSM8KDataLoader(split="train")
    >>> mmlu_loader = MMLUDataLoader(subset="abstract_algebra", split="test")
    >>>
    >>> # Create multiplexed dataloader
    >>> train_loader = create_multiplex_dataloader(
    ...     datasets=[
    ...         ("arc", arc_loader.load_data()),
    ...         ("gsm8k", gsm8k_loader.load_data()),
    ...         ("mmlu", mmlu_loader.load_data()),
    ...     ],
    ...     batch_size=32,
    ...     num_workers=4,
    ...     pin_memory=True,
    ...     prefetch_factor=2,
    ...     sampling_strategy="uniform"
    ... )
    >>>
    >>> for batch in train_loader:
    ...     # Each batch contains data from the mixture of datasets
    ...     print(batch)
"""

import random
from typing import Any, Callable, Dict, List, Optional, Tuple

import torch
from torch.utils.data import DataLoader, Dataset, Sampler


class MultiplexDataset(Dataset):
    """
    A PyTorch Dataset that multiplexes multiple datasets.

    This dataset wraps multiple source datasets and provides a unified interface
    to sample from them. It supports different sampling strategies to control
    how examples are selected from each source dataset.

    Attributes:
        datasets: List of (name, dataset) tuples
        sampling_strategy: How to sample from datasets ('uniform', 'weighted', 'proportional')
        sampling_weights: Optional weights for each dataset (used with 'weighted' strategy)
        shuffle_seed: Random seed for deterministic shuffling
    """

    def __init__(
        self,
        datasets: List[Tuple[str, List[Dict]]],
        sampling_strategy: str = "proportional",
        sampling_weights: Optional[List[float]] = None,
        shuffle_seed: int = 42,
    ):
        """
        Initialize the multiplex dataset.

        Args:
            datasets: List of (dataset_name, dataset_examples) tuples
            sampling_strategy: One of:
                - 'proportional': Sample proportionally to dataset size (default)
                - 'uniform': Sample uniformly from each dataset
                - 'weighted': Sample according to provided weights
            sampling_weights: Optional weights for 'weighted' strategy
            shuffle_seed: Random seed for reproducibility
        """
        super().__init__()

        assert len(datasets) > 0, "Must provide at least one dataset"
        assert sampling_strategy in [
            "proportional",
            "uniform",
            "weighted",
        ], f"sampling_strategy must be 'proportional', 'uniform', or 'weighted', got {sampling_strategy}"

        if sampling_strategy == "weighted":
            assert (
                sampling_weights is not None
            ), "Must provide sampling_weights when using 'weighted' strategy"
            assert len(sampling_weights) == len(
                datasets
            ), f"Number of weights ({len(sampling_weights)}) must match number of datasets ({len(datasets)})"

        self.datasets = datasets
        self.dataset_names = [name for name, _ in datasets]
        self.dataset_data = [data for _, data in datasets]
        self.dataset_lengths = [len(data) for data in self.dataset_data]
        self.sampling_strategy = sampling_strategy
        self.sampling_weights = sampling_weights
        self.shuffle_seed = shuffle_seed

        # Build unified index map: (dataset_idx, example_idx)
        self.index_map = []
        for dataset_idx, dataset_length in enumerate(self.dataset_lengths):
            for example_idx in range(dataset_length):
                self.index_map.append((dataset_idx, example_idx))

        # Total number of examples
        self.total_examples = len(self.index_map)

        # Calculate sampling probabilities based on strategy
        self._calculate_sampling_probabilities()

        # Shuffle index map according to sampling strategy
        self._shuffle_index_map()

    def _calculate_sampling_probabilities(self):
        """Calculate sampling probabilities for each dataset based on strategy."""
        if self.sampling_strategy == "proportional":
            # Proportional to dataset size
            total = sum(self.dataset_lengths)
            self.probabilities = [length / total for length in self.dataset_lengths]

        elif self.sampling_strategy == "uniform":
            # Equal probability for each dataset
            num_datasets = len(self.datasets)
            self.probabilities = [1.0 / num_datasets] * num_datasets

        elif self.sampling_strategy == "weighted":
            # Use provided weights (normalize them)
            total = sum(self.sampling_weights)
            self.probabilities = [w / total for w in self.sampling_weights]

    def _shuffle_index_map(self):
        """Shuffle the index map deterministically based on sampling strategy."""
        rng = random.Random(self.shuffle_seed)

        if self.sampling_strategy == "proportional":
            # Simple deterministic shuffle (natural mixing based on dataset sizes)
            rng.shuffle(self.index_map)

        elif self.sampling_strategy in ["uniform", "weighted"]:
            # For uniform/weighted, we want to ensure proper distribution
            # Create a weighted shuffled index map
            new_index_map = []
            dataset_indices = [list(range(length)) for length in self.dataset_lengths]

            # Shuffle each dataset's indices
            for indices in dataset_indices:
                rng.shuffle(indices)

            # Pointers for each dataset
            pointers = [0] * len(self.datasets)

            # Build new index map by sampling according to probabilities
            for _ in range(self.total_examples):
                # Choose dataset based on probabilities
                dataset_idx = rng.choices(
                    range(len(self.datasets)), weights=self.probabilities, k=1
                )[0]

                # Get next example from chosen dataset (with wraparound)
                if pointers[dataset_idx] >= self.dataset_lengths[dataset_idx]:
                    pointers[dataset_idx] = 0
                    rng.shuffle(dataset_indices[dataset_idx])

                example_idx = dataset_indices[dataset_idx][pointers[dataset_idx]]
                pointers[dataset_idx] += 1

                new_index_map.append((dataset_idx, example_idx))

            self.index_map = new_index_map

    def __len__(self) -> int:
        """Return total number of examples."""
        return self.total_examples

    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """
        Get example at index.

        Args:
            idx: Index of example to retrieve

        Returns:
            Dictionary containing the example data with added 'dataset_name' field
        """
        if idx < 0 or idx >= self.total_examples:
            raise IndexError(
                f"Index {idx} out of range for dataset with {self.total_examples} examples"
            )

        dataset_idx, example_idx = self.index_map[idx]
        example = self.dataset_data[dataset_idx][example_idx].copy()

        # Add metadata about which dataset this came from
        example["dataset_name"] = self.dataset_names[dataset_idx]
        example["dataset_idx"] = dataset_idx

        return example

    def get_dataset_stats(self) -> Dict[str, Any]:
        """
        Get statistics about the multiplex dataset.

        Returns:
            Dictionary with dataset statistics
        """
        stats = {
            "num_datasets": len(self.datasets),
            "total_examples": self.total_examples,
            "sampling_strategy": self.sampling_strategy,
            "datasets": [],
        }

        for name, length, prob in zip(
            self.dataset_names, self.dataset_lengths, self.probabilities
        ):
            stats["datasets"].append(
                {
                    "name": name,
                    "num_examples": length,
                    "sampling_probability": prob,
                    "percentage": f"{100 * prob:.2f}%",
                }
            )

        return stats


class EpochSampler(Sampler):
    """
    A sampler that iterates through the dataset once per epoch.

    This sampler ensures that each example in the dataset is seen exactly once
    per epoch, which is useful for training with a fixed number of epochs.
    """

    def __init__(self, dataset: Dataset, shuffle: bool = True, seed: int = 0):
        """
        Initialize the epoch sampler.

        Args:
            dataset: The dataset to sample from
            shuffle: Whether to shuffle indices each epoch
            seed: Random seed for shuffling
        """
        self.dataset = dataset
        self.shuffle = shuffle
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        """Iterate through indices."""
        indices = list(range(len(self.dataset)))

        if self.shuffle:
            # Use epoch-based seed for different shuffle each epoch
            rng = torch.Generator()
            rng.manual_seed(self.seed + self.epoch)
            indices = torch.randperm(len(self.dataset), generator=rng).tolist()

        yield from indices

    def __len__(self):
        """Return dataset length."""
        return len(self.dataset)

    def set_epoch(self, epoch: int):
        """Set the current epoch (for different shuffle each epoch)."""
        self.epoch = epoch


def default_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Default collate function for multiplex datasets.

    This function batches together a list of examples into a single batch dictionary.
    It handles different types of data appropriately.

    Args:
        batch: List of example dictionaries

    Returns:
        Batched dictionary with lists of values for each key
    """
    if len(batch) == 0:
        return {}

    # Collect all keys from all examples
    all_keys = set()
    for example in batch:
        all_keys.update(example.keys())

    batched = {}
    for key in all_keys:
        values = [example.get(key) for example in batch]

        # Try to stack tensors if all values are tensors
        if all(isinstance(v, torch.Tensor) for v in values if v is not None):
            try:
                batched[key] = torch.stack([v for v in values if v is not None])
            except (RuntimeError, TypeError):
                # If stacking fails, keep as list
                batched[key] = values
        else:
            # Keep as list for non-tensor data
            batched[key] = values

    return batched


def create_sft_collate_fn(
    tokenizer, device: torch.device, pad_token_id: Optional[int] = None
):
    """
    Create a collate function for SFT training that tokenizes conversations and creates masks.

    This mimics the behavior of chat_sft.py's collate_and_yield function.

    Args:
        tokenizer: Tokenizer with encode method (from get_custom_tokenizer)
        device: Device to move tensors to (cuda/cpu)
        pad_token_id: Token ID for padding (default: uses <|assistant_end|>)

    Returns:
        Collate function that takes a batch and returns (inputs, targets) tensors

    Example:
        >>> from gpt_2.utils import get_custom_tokenizer
        >>> tokenizer, _ = get_custom_tokenizer()
        >>> device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        >>> collate_fn = create_sft_collate_fn(tokenizer, device)
        >>>
        >>> dataloader = create_multiplex_dataloader(
        ...     datasets=datasets,
        ...     collate_fn=collate_fn,
        ...     batch_size=32,
        ... )
        >>>
        >>> for inputs, targets in dataloader:
        ...     loss = model(inputs, targets)
    """
    if pad_token_id is None:
        # Use the token ID for <|assistant_end|>
        pad_token_id = tokenizer.encode("<|assistant_end|>", allowed_special="all")[0]

    def collate_fn(batch: List[Dict[str, Any]]) -> tuple:
        """
        Collate a batch of conversations into padded input/target tensors.

        Args:
            batch: List of example dicts with 'messages' key

        Returns:
            Tuple of (inputs, targets) tensors ready for training
        """
        # Import here to avoid circular dependency
        from eval_tasks.chat_core.utils import render_conversation_for_training

        # Tokenize all conversations
        tokenized_batch = []
        for example in batch:
            # Render conversation with tokens and mask
            ids, mask = render_conversation_for_training(example, tokenizer)
            tokenized_batch.append((ids, mask))

        # Find max length in batch
        nrows = len(tokenized_batch)
        ncols = (
            max(len(ids) for ids, mask in tokenized_batch) - 1
        )  # seq of n creates inputs/targets of n-1

        # Create padded tensors
        inputs = torch.full((nrows, ncols), pad_token_id, dtype=torch.long)
        targets = torch.full((nrows, ncols), -1, dtype=torch.long)  # -1 is ignore index

        # Fill in the data
        for i, (ids, mask) in enumerate(tokenized_batch):
            n = len(ids)
            ids_tensor = torch.tensor(ids, dtype=torch.long)
            inputs[i, : n - 1] = ids_tensor[:-1]

            # Apply mask to targets (mask out where mask is 0)
            row_targets = ids_tensor[1:]
            mask_tensor = torch.tensor(mask[1:], dtype=torch.long)
            row_targets[mask_tensor == 0] = -1  # Mask out targets where mask is 0
            targets[i, : n - 1] = row_targets

        # Move to device
        inputs = inputs.to(device)
        targets = targets.to(device)

        return inputs, targets

    return collate_fn


def create_multiplex_dataloader(
    datasets: List[Tuple[str, List[Dict]]],
    batch_size: int = 32,
    shuffle: bool = True,
    num_workers: int = 0,
    pin_memory: bool = False,
    prefetch_factor: Optional[int] = None,
    persistent_workers: bool = False,
    drop_last: bool = False,
    sampling_strategy: str = "proportional",
    sampling_weights: Optional[List[float]] = None,
    collate_fn: Optional[Callable] = None,
    shuffle_seed: int = 42,
) -> DataLoader:
    """
    Create a PyTorch DataLoader that multiplexes multiple datasets.

    This function creates a DataLoader with all PyTorch features (pin_memory,
    prefetch_factor, num_workers, etc.) that samples from multiple datasets
    according to a specified strategy.

    Args:
        datasets: List of (dataset_name, dataset_examples) tuples
        batch_size: Number of examples per batch
        shuffle: Whether to shuffle data each epoch
        num_workers: Number of worker processes for data loading
        pin_memory: Whether to pin memory (useful for GPU training)
        prefetch_factor: Number of batches to prefetch per worker (None = default 2)
        persistent_workers: Keep workers alive between epochs
        drop_last: Drop last incomplete batch
        sampling_strategy: How to sample from datasets ('uniform', 'weighted', 'proportional')
        sampling_weights: Optional weights for each dataset (used with 'weighted' strategy)
        collate_fn: Custom collate function (default: default_collate_fn)
        shuffle_seed: Random seed for reproducibility

    Returns:
        PyTorch DataLoader that yields batches from the multiplexed datasets

    Example:
        >>> datasets = [
        ...     ("arc", arc_loader.load_data()),
        ...     ("gsm8k", gsm8k_loader.load_data()),
        ... ]
        >>> loader = create_multiplex_dataloader(
        ...     datasets=datasets,
        ...     batch_size=32,
        ...     num_workers=4,
        ...     pin_memory=True,
        ...     prefetch_factor=2
        ... )
    """
    # Create the multiplex dataset
    dataset = MultiplexDataset(
        datasets=datasets,
        sampling_strategy=sampling_strategy,
        sampling_weights=sampling_weights,
        shuffle_seed=shuffle_seed,
    )

    # Use default collate function if none provided
    if collate_fn is None:
        collate_fn = default_collate_fn

    # Build DataLoader kwargs
    dataloader_kwargs = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "num_workers": num_workers,
        "pin_memory": pin_memory,
        "drop_last": drop_last,
        "collate_fn": collate_fn,
    }

    # Add prefetch_factor only if num_workers > 0 (it's not valid for num_workers=0)
    if num_workers > 0:
        if prefetch_factor is not None:
            dataloader_kwargs["prefetch_factor"] = prefetch_factor
        if persistent_workers:
            dataloader_kwargs["persistent_workers"] = persistent_workers

    # Create and return DataLoader
    dataloader = DataLoader(dataset, **dataloader_kwargs)

    # Attach dataset stats for easy access
    dataloader.dataset_stats = dataset.get_dataset_stats()

    return dataloader


def print_dataloader_stats(dataloader: DataLoader):
    """
    Print statistics about a multiplex dataloader.

    Args:
        dataloader: The multiplex dataloader to print stats for
    """
    if not hasattr(dataloader, "dataset_stats"):
        print("No dataset stats available for this dataloader")
        return

    stats = dataloader.dataset_stats
    print(f"\n{'='*70}")
    print("Multiplex DataLoader Statistics")
    print(f"{'='*70}")
    print(f"Number of datasets: {stats['num_datasets']}")
    print(f"Total examples: {stats['total_examples']:,}")
    print(f"Sampling strategy: {stats['sampling_strategy']}")
    print("\nDataset Breakdown:")
    print(f"{'-'*70}")
    print(f"{'Dataset':<30} {'Examples':<15} {'Sampling %':<15}")
    print(f"{'-'*70}")

    for ds in stats["datasets"]:
        print(f"{ds['name']:<30} {ds['num_examples']:<15,} {ds['percentage']:<15}")

    print(f"{'='*70}\n")


# Example usage
if __name__ == "__main__":

    from dataloaders.arc_dataloader import ARCDataLoader
    from dataloaders.gsm8k_dataloader import GSM8KDataLoader
    from dataloaders.mmlu_dataloader import MMLUDataLoader
    from dataloaders.simplespelling_dataloader import SimpleSpellingDataLoader
    from dataloaders.spellingbee_dataloader import SpellingBeeDataLoader
    from gpt_2.config import GPTConfig
    from gpt_2.utils import get_custom_tokenizer

    print("\n" + "=" * 80)
    print("Multiplex DataLoader - Real Dataset Example")
    print("=" * 80 + "\n")

    # Cache directory for HuggingFace datasets
    config = GPTConfig()
    cache_dir = config.chat_core_hf_cache_dir

    # Load actual datasets (small samples for demo)
    # Use format_as_conversation=True to get data in conversation format directly
    print("Loading datasets...")
    arc_data = ARCDataLoader(
        subset="ARC-Easy", split="train", cache_dir=cache_dir
    ).load_data(max_examples=50, format_as_conversation=True)
    gsm8k_data = GSM8KDataLoader(split="train", cache_dir=cache_dir).load_data(
        max_examples=50, format_as_conversation=True
    )
    mmlu_data = MMLUDataLoader(
        subset="auxiliary_train", split="train", cache_dir=cache_dir
    ).load_data(max_examples=30, format_as_conversation=True)
    # SpellingBee and SimpleSpelling already return conversation format by default
    spelling_bee_data = SpellingBeeDataLoader(
        size=30, split="train", cache_dir=cache_dir
    ).load_data()
    simple_spelling_data = SimpleSpellingDataLoader(
        size=20, split="train", cache_dir=cache_dir
    ).load_data()

    print(f"✓ Loaded {len(arc_data)} ARC examples")
    print(f"✓ Loaded {len(gsm8k_data)} GSM8K examples")
    print(f"✓ Loaded {len(mmlu_data)} MMLU examples")
    print(f"✓ Loaded {len(spelling_bee_data)} SpellingBee examples")
    print(f"✓ Loaded {len(simple_spelling_data)} SimpleSpelling examples\n")

    # Example 1: Proportional sampling (default)
    print("=" * 80)
    print("Example 1: Proportional Sampling")
    print("=" * 80 + "\n")

    enc, _ = get_custom_tokenizer()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    collate_fn = create_sft_collate_fn(enc, device)

    dataloader = create_multiplex_dataloader(
        datasets=[
            ("arc", arc_data),
            ("gsm8k", gsm8k_data),
            ("mmlu", mmlu_data),
            ("spelling_bee", spelling_bee_data),
            ("simple_spelling", simple_spelling_data),
        ],
        batch_size=8,
        shuffle=True,
        num_workers=0,
        sampling_strategy="uniform",
        collate_fn=collate_fn,
    )

    print_dataloader_stats(dataloader)

    # Show first batch (collated and tokenized)
    print("First batch sample (tokenized):")
    batch = next(iter(dataloader))
    inputs, targets = batch
    print(f"  Batch size: {inputs.shape[0]}")
    print(f"  Sequence length: {inputs.shape[1]}")
    print(f"  Inputs shape: {inputs.shape}")
    print(f"  Targets shape: {targets.shape}")
    print(f"  Device: {inputs.device}")

    # Show dataset distribution (create a dataloader without collate_fn to see raw data)
    print("\nDataset distribution in first 20 batches:")
    raw_dataloader = create_multiplex_dataloader(
        datasets=[
            ("arc", arc_data),
            ("gsm8k", gsm8k_data),
            ("mmlu", mmlu_data),
            ("spelling_bee", spelling_bee_data),
            ("simple_spelling", simple_spelling_data),
        ],
        batch_size=8,
        shuffle=True,
        num_workers=0,
        sampling_strategy="uniform",
    )
    dataset_counts = {}
    for i, batch in enumerate(raw_dataloader):
        if i >= 20:
            break
        # PyTorch's default collate converts list of dicts to dict of lists
        # So batch is a dict with 'dataset_name' key containing a list
        if isinstance(batch, dict) and "dataset_name" in batch:
            for name in batch["dataset_name"]:
                dataset_counts[name] = dataset_counts.get(name, 0) + 1
        else:
            # Fallback: iterate through batch as list of dicts
            for item in batch:
                name = item.get("dataset_name", "unknown")
                dataset_counts[name] = dataset_counts.get(name, 0) + 1

    total = sum(dataset_counts.values())
    for name, count in sorted(dataset_counts.items()):
        percentage = (count / total) * 100
        print(f"  {name:<20} {count:>3} ({percentage:>5.1f}%)")

    # # Example 2: Uniform sampling
    # print("\n" + "="*80)
    # print("Example 2: Uniform Sampling")
    # print("="*80 + "\n")

    # dataloader_uniform = create_multiplex_dataloader(
    #     datasets=[
    #         ("arc", arc_data),
    #         ("gsm8k", gsm8k_data),
    #         ("mmlu", mmlu_data),
    #     ],
    #     batch_size=8,
    #     shuffle=True,
    #     num_workers=0,
    #     sampling_strategy="uniform",
    # )

    # print_dataloader_stats(dataloader_uniform)

    # # Count distribution in first 20 batches
    # dataset_counts = {}
    # for i, batch in enumerate(dataloader_uniform):
    #     if i >= 20:
    #         break
    #     for name in batch["dataset_name"]:
    #         dataset_counts[name] = dataset_counts.get(name, 0) + 1

    # total = sum(dataset_counts.values())
    # print("Distribution in first 20 batches:")
    # for name, count in sorted(dataset_counts.items()):
    #     percentage = (count / total) * 100
    #     print(f"  {name:<20} {count:>3} ({percentage:>5.1f}%)")

    # # Example 3: Show a complete example with data
    # print("\n" + "="*80)
    # print("Example 3: Sample Data from Batch")
    # print("="*80 + "\n")

    # batch = next(iter(dataloader))
    # print(f"Batch contains {len(batch['dataset_name'])} examples\n")

    # # Show first example from batch
    # print("First example from batch:")
    # print(json.dumps({
    #     "dataset": batch['dataset_name'][0],
    #     "messages": batch['messages'][0]
    # }, indent=2, ensure_ascii=False))

    # print("\n" + "="*80)
    # print("Demo Complete!")
    # print("="*80 + "\n")
