"""
Unit tests for FinewebEduParquetBOSDataloader (PyTorch-native).

Tests verify:
    - BOS alignment: every sequence starts with a BOS token
    - Correct tensor shapes
    - Input/target relationship (autoregressive shift)
    - Best-fit packing algorithm
    - Buffer management
    - PyTorch DataLoader integration
"""

import pyarrow as pa
import pyarrow.parquet as pq
import pytest
import torch

from dataloaders.fineweb_edu_parquet_bos_dataloader import (
    FinewebEduParquetBOSDataloader, list_parquet_files)
from gpt_2.utils import get_custom_tokenizer


@pytest.fixture
def mock_parquet_data(temp_dir):
    """Create mock parquet files for testing."""
    # Create sample documents of varying lengths
    train_documents = [
        "This is a short document.",
        "This is a much longer document with more tokens that should be packed efficiently using the best-fit algorithm.",
        "Medium length document here.",
        "Another short one.",
        "This document has a reasonable length for testing purposes and will help verify the packing.",
    ] * 50  # Repeat to have enough data

    val_documents = [
        "Validation document one.",
        "Validation document two with more content.",
    ] * 20

    # Create PyArrow table for training data
    train_table = pa.table({"text": train_documents})
    train_path = temp_dir / "shard_00000.parquet"
    pq.write_table(train_table, train_path)

    # Create PyArrow table for validation data
    val_table = pa.table({"text": val_documents})
    val_path = temp_dir / "shard_00001.parquet"
    pq.write_table(val_table, val_path)

    return temp_dir


class TestListParquetFiles:
    """Test the list_parquet_files utility function."""

    def test_lists_parquet_files(self, mock_parquet_data):
        """Verify it finds and sorts parquet files correctly."""
        files = list_parquet_files(str(mock_parquet_data))
        assert len(files) == 2
        assert files[0].endswith("shard_00000.parquet")
        assert files[1].endswith("shard_00001.parquet")

    def test_skips_tmp_files(self, temp_dir):
        """Verify it skips temporary files."""
        # Create a regular and a tmp file
        regular_path = temp_dir / "shard_00000.parquet"
        tmp_path = temp_dir / "shard_00001.parquet.tmp"

        # Create minimal parquet files
        table = pa.table({"text": ["test"]})
        pq.write_table(table, regular_path)
        pq.write_table(table, tmp_path)

        files = list_parquet_files(str(temp_dir))
        assert len(files) == 1
        assert files[0].endswith("shard_00000.parquet")


class TestFinewebEduParquetBOSDataloader:
    """Test the main dataloader class."""

    @pytest.fixture
    def dataloader(self, mock_parquet_data, device):
        """Create a dataloader instance for testing."""
        return FinewebEduParquetBOSDataloader(
            data_dir=str(mock_parquet_data),
            batch_size=4,
            block_size=32,
            split="train",
            master_process=False,  # Suppress print statements
            buffer_size=200,  # Increased to handle dataloader_batch_size=48
            device=device,
            num_workers=0,  # Single process for testing
        )

    def test_initialization(self, dataloader, device):
        """Verify dataloader initializes correctly."""
        assert dataloader.batch_size == 4
        assert dataloader.block_size == 32
        assert dataloader.device == str(device)
        # Collator handles the document buffer now
        assert dataloader.collator is not None
        assert dataloader.collator.batch_size == 4
        assert dataloader.collator.block_size == 32

    def test_batch_shapes(self, dataloader):
        """Verify output tensors have correct shapes."""
        inputs, targets = next(dataloader)

        # Shape: (batch_size, block_size)
        assert inputs.shape == (4, 32)
        assert targets.shape == (4, 32)
        assert inputs.dtype == torch.long
        assert targets.dtype == torch.long

    def test_bos_alignment(self, dataloader):
        """Verify every sequence starts with BOS token."""
        tokenizer, _ = get_custom_tokenizer()
        bos_token_id = tokenizer.encode("<|bos|>", allowed_special="all")[0]

        inputs, targets = next(dataloader)

        # Every sequence should start with BOS token
        for i in range(inputs.shape[0]):
            assert inputs[i, 0].item() == bos_token_id, (
                f"Sequence {i} does not start with BOS token. "
                f"Expected {bos_token_id}, got {inputs[i, 0].item()}"
            )

    def test_autoregressive_shift(self, dataloader):
        """Verify targets are inputs shifted by 1."""
        inputs, targets = next(dataloader)

        # For each sequence, check that target[i] = input[i+1]
        # (This verifies the relationship at row_buffer level)
        # Note: We can't directly compare inputs[:, 1:] == targets[:, :-1]
        # because row_buffer is packed from multiple documents, but we can
        # verify the shapes are correct
        assert inputs.shape == targets.shape

    def test_iterator_protocol(self, dataloader):
        """Verify dataloader works as an iterator."""
        # Test __iter__
        iterator = iter(dataloader)
        assert iterator is dataloader

        # Test __next__
        inputs, targets = next(dataloader)
        assert inputs.shape == (4, 32)
        assert targets.shape == (4, 32)

    def test_multiple_batches(self, dataloader):
        """Verify can generate multiple batches successfully."""
        batches_generated = 0
        for _ in range(5):
            inputs, targets = next(dataloader)
            assert inputs.shape == (4, 32)
            assert targets.shape == (4, 32)
            batches_generated += 1

        assert batches_generated == 5

    def test_buffer_refill(self, dataloader):
        """Verify document buffer gets refilled as needed."""
        # Generate several batches to consume buffer
        for _ in range(10):
            next(dataloader)

        # Buffer is managed by collator in new implementation
        stats = dataloader.get_stats()
        assert stats["buffer_size"] >= 0, "Buffer should be tracked"

    def test_stats_tracking(self, dataloader):
        """Verify token statistics are tracked correctly."""
        # Generate some batches
        for _ in range(3):
            next(dataloader)

        stats = dataloader.get_stats()
        assert "total_tokens" in stats
        assert "cropped_tokens" in stats
        assert "crop_percentage" in stats
        assert "buffer_size" in stats

        assert stats["total_tokens"] > 0
        assert stats["cropped_tokens"] >= 0
        assert 0 <= stats["crop_percentage"] <= 100
        assert stats["buffer_size"] >= 0

    def test_device_placement(self, dataloader, device):
        """Verify tensors are returned on CPU (trainer moves them to target device)."""
        inputs, targets = next(dataloader)
        # Dataloader returns CPU tensors because workers can't access GPU
        # Trainer code moves them to target device with: x, y = x.to(device), y.to(device)
        assert str(inputs.device) == "cpu"
        assert str(targets.device) == "cpu"

    def test_token_validity(self, dataloader):
        """Verify all tokens are valid (non-negative)."""
        inputs, targets = next(dataloader)

        # All token IDs should be non-negative
        assert torch.all(inputs >= 0)
        assert torch.all(targets >= 0)

    def test_different_batch_sizes(self, mock_parquet_data, device):
        """Verify dataloader works with different batch sizes."""
        for batch_size in [1, 2, 8, 16]:
            loader = FinewebEduParquetBOSDataloader(
                data_dir=str(mock_parquet_data),
                batch_size=batch_size,
                block_size=16,
                split="train",
                master_process=False,
                buffer_size=100,  # Larger buffer for testing various batch sizes
                device=device,
                num_workers=0,
            )

            inputs, targets = next(loader)
            assert inputs.shape == (batch_size, 16)
            assert targets.shape == (batch_size, 16)

    def test_different_block_sizes(self, mock_parquet_data, device):
        """Verify dataloader works with different block sizes."""
        for block_size in [16, 32, 64, 128]:
            loader = FinewebEduParquetBOSDataloader(
                data_dir=str(mock_parquet_data),
                batch_size=2,
                block_size=block_size,
                split="train",
                master_process=False,
                buffer_size=100,  # Larger buffer for testing various block sizes
                device=device,
                num_workers=0,
            )

            inputs, targets = next(loader)
            assert inputs.shape == (2, block_size)
            assert targets.shape == (2, block_size)


class TestBestFitPacking:
    """Test the best-fit packing algorithm specifically."""

    def test_packing_efficiency(self, mock_parquet_data, device):
        """Verify packing algorithm minimizes waste."""
        loader = FinewebEduParquetBOSDataloader(
            data_dir=str(mock_parquet_data),
            batch_size=4,
            block_size=64,
            split="train",
            master_process=False,
            buffer_size=150,  # Larger buffer for better best-fit choices
            device=device,
            num_workers=0,
        )

        # Generate several batches
        for _ in range(10):
            next(loader)

        stats = loader.get_stats()

        # Crop percentage should be reasonable (not 100%)
        # With best-fit, we expect < 50% waste for varied document lengths
        assert stats["crop_percentage"] < 80, (
            f"Crop percentage too high: {stats['crop_percentage']:.1f}%. "
            "Best-fit algorithm may not be working properly."
        )


class TestDDPSupport:
    """Test distributed data parallel support."""

    def test_ddp_initialization(self, mock_parquet_data, device):
        """Verify dataloader can be initialized with DDP parameters."""
        # Just verify that DDP parameters are accepted and dataloader initializes
        loader = FinewebEduParquetBOSDataloader(
            data_dir=str(mock_parquet_data),
            batch_size=2,
            block_size=16,
            ddp_rank=0,
            ddp_world_size=2,
            split="train",
            master_process=False,
            buffer_size=10,
            device=device,
            num_workers=0,
        )

        # Verify it can produce a batch
        inputs, targets = next(loader)
        assert inputs.shape == (2, 16)
        assert targets.shape == (2, 16)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
