import os
from tqdm import tqdm
import numpy as np
import tiktoken
from datasets import load_dataset

# Number of workers for parallel processing (typically ~half of CPU cores)
num_proc = 8

# Number of workers for dataset loading (may differ based on network speed)
num_proc_load_dataset = num_proc

# Initialize GPT-2 tokenizer
enc = tiktoken.get_encoding("gpt2")

if __name__ == "__main__":
    # Load OpenWebText dataset (~54GB, 8M documents)
    cache_dir = "/sensei-fs/users/divgoyal/openwebtext/hf_cache"
    dataset = load_dataset(
        "Skylion007/openwebtext", num_proc=num_proc_load_dataset, cache_dir=cache_dir
    )

    # Create train/val splits (0.05% for validation)
    # Result: ~8M train examples, ~4K validation examples
    split_dataset = dataset["train"].train_test_split(
        test_size=0.0005, seed=2357, shuffle=True
    )
    split_dataset["val"] = split_dataset.pop("test")

    # Tokenize dataset using GPT-2 BPE encoding
    def process(example):
        ids = enc.encode_ordinary(example["text"])  # Encode text without special tokens
        ids.append(enc.eot_token)  # Append end-of-text token (50256)
        return {"ids": ids, "len": len(ids)}

    tokenized = split_dataset.map(
        process,
        remove_columns=["text"],
        desc="tokenizing the splits",
        num_proc=num_proc,
    )

    # Write tokenized data to binary files for training
    # Strategy: Concatenate all token IDs into one continuous array per split
    output_dir = "/sensei-fs/users/divgoyal/openwebtext"
    os.makedirs(output_dir, exist_ok=True)

    for split, dset in tokenized.items():
        # Calculate total number of tokens across all documents
        arr_len = np.sum(dset["len"], dtype=np.uint64)

        # Create memory-mapped file (acts like numpy array without loading into RAM)
        filename = os.path.join(output_dir, f"{split}.bin")
        dtype = np.uint16  # uint16 sufficient since max token value is 50256
        arr = np.memmap(filename, dtype=dtype, mode="w+", shape=(arr_len,))

        # Process in batches to avoid memory overflow
        total_batches = 1024
        idx = 0
        for batch_idx in tqdm(range(total_batches), desc=f"writing {filename}"):
            # Get one batch (1/1024th of dataset)
            batch = dset.shard(
                num_shards=total_batches, index=batch_idx, contiguous=True
            ).with_format("numpy")

            # Concatenate all token IDs in this batch: [[1,2,3], [4,5]] -> [1,2,3,4,5]
            arr_batch = np.concatenate(batch["ids"])

            # Write batch to file and advance position
            arr[idx : idx + len(arr_batch)] = arr_batch
            idx += len(arr_batch)

        # Ensure all data is written to disk
        arr.flush()

    # Output files:
    # train.bin: ~17GB, ~9B tokens
    # val.bin:   ~8.5MB, ~4M tokens
