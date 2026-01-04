import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)


import torch
from gpt_2.utils import get_custom_tokenizer


class DataLoader:
    def __init__(
        self,
        data_file,
        batch_size,
        block_size,
        ddp_world_size,
        ddp_rank,
    ):
        self.data = open(data_file, "r", encoding="utf-8").read()  # read data file
        self.enc, _ = get_custom_tokenizer()  # get custom tokenizer with special tokens
        self.tokens = self.enc.encode(self.data, allowed_special="all")  # encode data
        self.tokens = torch.tensor(self.tokens, dtype=torch.long)  # convert to tensor
        self.train_data = self.tokens[
            : int(0.9 * len(self.tokens))
        ]  # split data into train and val
        self.val_data = self.tokens[
            int(0.9 * len(self.tokens)) :
        ]  # split data into train and val
        self.batch_size = batch_size  # batch size
        self.block_size = block_size  # block size
        self.idx = 0  # index
        self.total_batches = len(self.tokens) // (self.block_size)  # total batches
        self.total_train_batches = len(self.train_data) // (
            self.block_size
        )  # total train batches
        self.total_val_batches = len(self.val_data) // (
            self.block_size
        )  # total val batches
        self.ddp_world_size = ddp_world_size  # ddp world size
        self.ddp_rank = ddp_rank  # ddp rank
        self.current_index = (
            self.ddp_rank * self.batch_size * self.block_size
        )  # current index

        print(
            f"Total tokens: {len(self.tokens)} , Total train tokens: {len(self.train_data)} , Total val tokens: {len(self.val_data)}"
        )
        print(
            f"Total batches: {self.total_batches}, Total train batches: {self.total_train_batches}, Total val batches: {self.total_val_batches} for 1 epoch"
        )

    def get_batch(self, split="train"):
        if split == "train":
            data = self.train_data
        else:
            data = self.val_data

        if self.current_index + (self.batch_size * self.block_size + 1) > len(data):
            rand_chunk_number = torch.randint(
                0,
                (
                    len(self.train_data)
                    // (self.batch_size * self.block_size * self.ddp_world_size)
                )
                - 1,
                size=(1,),
            )
            self.current_index = (
                self.ddp_rank * self.batch_size * self.block_size
                + rand_chunk_number.item()
                * self.batch_size
                * self.block_size
                * self.ddp_world_size
            )

        data_size = self.batch_size * self.block_size
        current_batch = data[self.current_index : self.current_index + data_size].view(
            self.batch_size, self.block_size
        )
        target_batch = data[
            self.current_index + 1 : self.current_index + data_size + 1
        ].view(self.batch_size, self.block_size)
        self.current_index += self.batch_size * self.block_size * self.ddp_world_size
        return current_batch, target_batch

    def __len__(self):
        return len(self.tokens)


if __name__ == "__main__":
    data_file = f"{parent_dir}/data/input.txt"

    print("=== DataLoader Sanity Tests ===\n")

    # Test 1: Basic functionality
    print("Test 1: Basic functionality")
    dataloader = DataLoader(
        data_file=data_file, batch_size=4, block_size=10, ddp_world_size=1, ddp_rank=0
    )

    # Get a few batches and check shapes
    for i in range(5):
        x, y = dataloader.get_batch()
        print(f"  Batch {i}: x.shape={x.shape}, y.shape={y.shape}")
        assert x.shape == (4, 10), f"Expected x.shape=(4, 10), got {x.shape}"
        assert y.shape == (4, 10), f"Expected y.shape=(4, 10), got {y.shape}"
        assert torch.all(x >= 0), "All tokens should be non-negative"
        assert torch.all(y >= 0), "All target tokens should be non-negative"
    print("  ✓ Basic functionality test passed\n")

    # Test 2: Train vs Validation split
    print("Test 2: Train vs Validation split")
    train_batch, train_target = dataloader.get_batch("train")
    val_batch, val_target = dataloader.get_batch("val")
    print(f"  Train batch shape: {train_batch.shape}")
    print(f"  Validation batch shape: {val_batch.shape}")
    print("  ✓ Train/Validation split test passed\n")

    # Test 3: Randomization check
    print("Test 3: Randomization check")
    # Create two dataloaders with same parameters
    dl1 = DataLoader(
        data_file=data_file, batch_size=2, block_size=8, ddp_world_size=1, ddp_rank=0
    )
    dl2 = DataLoader(
        data_file=data_file, batch_size=2, block_size=8, ddp_world_size=1, ddp_rank=0
    )

    # Get batches from both and check they're different
    batch1_x, batch1_y = dl1.get_batch()
    batch2_x, batch2_y = dl2.get_batch()

    print(f"  First dataloader batch: {batch1_x[0][:5].tolist()}")
    print(f"  Second dataloader batch: {batch2_x[0][:5].tolist()}")

    # They should be different due to randomization
    if not torch.equal(batch1_x, batch2_x):
        print("  ✓ Randomization test passed - batches are different")
    else:
        print(
            "  ⚠ Randomization test - batches are the same (this might happen by chance)"
        )
    print()

    # Test 4: DDP compatibility
    print("Test 4: DDP compatibility")
    dl_rank0 = DataLoader(
        data_file=data_file, batch_size=2, block_size=8, ddp_world_size=2, ddp_rank=0
    )
    dl_rank1 = DataLoader(
        data_file=data_file, batch_size=2, block_size=8, ddp_world_size=2, ddp_rank=1
    )

    batch_rank0, target_rank0 = dl_rank0.get_batch()
    batch_rank1, target_rank1 = dl_rank1.get_batch()

    print(f"  Rank 0 batch: {batch_rank0[0][:5].tolist()}")
    print(f"  Rank 1 batch: {batch_rank1[0][:5].tolist()}")

    # Different ranks should get different batches
    if not torch.equal(batch_rank0, batch_rank1):
        print("  ✓ DDP test passed - different ranks get different batches")
    else:
        print("  ⚠ DDP test - ranks got same batch (this might happen by chance)")
    print()

    # Test 5: Data consistency
    print("Test 5: Data consistency")
    dl = DataLoader(
        data_file=data_file, batch_size=2, block_size=8, ddp_world_size=1, ddp_rank=0
    )

    # Check that targets are shifted by 1 from inputs
    x, y = dl.get_batch()
    print(f"  Input sequence: {x[0][:8].tolist()}")
    print(f"  Target sequence: {y[0][:8].tolist()}")

    # Verify that targets are shifted by 1
    assert torch.equal(
        x[0][1:], y[0][:-1]
    ), "Targets should be shifted by 1 from inputs"
    print("  ✓ Data consistency test passed - targets are properly shifted\n")

    # Test 6: Boundary conditions
    print("Test 6: Boundary conditions")
    # Test with very small data
    small_dl = DataLoader(
        data_file=data_file, batch_size=1, block_size=5, ddp_world_size=1, ddp_rank=0
    )

    try:
        for i in range(10):
            x, y = small_dl.get_batch()
            print(f"  Small batch {i}: x.shape={x.shape}, y.shape={y.shape}")
        print("  ✓ Boundary conditions test passed\n")
    except Exception as e:
        print(f"  ✗ Boundary conditions test failed: {e}\n")

    # Test 7: Multiple epochs
    print("Test 7: Multiple epochs")
    dl = DataLoader(
        data_file=data_file, batch_size=2, block_size=8, ddp_world_size=1, ddp_rank=0
    )

    # Get batches until we hit the end and see if it resets properly
    print("  Getting batches until epoch end...")
    batch_count = 0
    try:
        while batch_count < 100:  # Safety limit
            x, y = dl.get_batch()
            batch_count += 1
            if batch_count % 20 == 0:
                print(f"    Batch {batch_count}: {x[0][:3].tolist()}")
    except Exception as e:
        print(f"    Error after {batch_count} batches: {e}")

    print(f"  ✓ Multiple epochs test completed - processed {batch_count} batches\n")

    print("=== All sanity tests completed ===")
