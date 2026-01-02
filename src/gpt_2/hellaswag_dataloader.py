import os
import json
import torch
import torch.nn.functional as F
import numpy as np


class HellaSwagDataloader:

    def __init__(
        self,
        data_dir,
        split="validation",
        ddp_world_size=1,
        ddp_rank=0,
        master_process=True,
        batch_size=128,
    ):
        """
        Initialize the HellaSwag data loader.

        Args:
            data_dir (str): Directory containing validation.json file.
            split (str, optional): Dataset split to use. Defaults to "validation".
            ddp_world_size (int, optional): Total number of processes in
                distributed evaluation. Defaults to 1.
            ddp_rank (int, optional): Rank of current process in distributed
                evaluation. Defaults to 0.
            master_process (bool, optional): Whether this is the master process.
                Controls logging output. Defaults to True.

        Raises:
            AssertionError: If the JSON file is not found in the data directory.
        """
        self.data_dir = data_dir
        self.split = split
        self.ddp_world_size = ddp_world_size
        self.ddp_rank = ddp_rank
        self.batch_size = batch_size

        # Load the JSON file
        data_file = os.path.join(data_dir, f"{split}.json")
        assert os.path.exists(data_file), f"Data file not found: {data_file}"

        if master_process:
            print(f"Loading HellaSwag {split} data from {data_file}")

        with open(data_file, "r") as f:
            self.examples = json.load(f)

        self.num_total_examples = len(self.examples)

        if master_process:
            print(f"Total number of examples: {self.num_total_examples} ")

        # Initialize position tracker for batching
        self.reset()

    def reset(self):
        """
        Reset the data loader to the beginning.

        This resets the current position to the beginning of the examples list for the current process.
        """
        self.current_position = self.ddp_rank * self.batch_size

    def next_batch(self):
        batch = self.examples[
            self.current_position : self.current_position + self.batch_size
        ]
        self.current_position += self.batch_size * self.ddp_world_size
        if self.current_position + (self.batch_size * self.ddp_world_size) > len(
            self.examples
        ):
            # Reset to this rank's starting position, not 0
            self.current_position = self.ddp_rank * self.batch_size

        # Process batch
        # Each example has 4 completions, so we'll have batch_size * 4 sequences
        max_length = 1024
        x = []
        y = []

        for example in batch:
            context_len = len(example["context_tokens"])

            # Concatenate context with each of the 4 endings
            for i in range(4):
                seq = example["context_tokens"] + example["ending_tokens"][i]

                # Truncate ending if too long (context is always <= 1024)
                if len(seq) > max_length:
                    # Keep all context + truncate ending to fit within 1024
                    seq = seq[:max_length]

                # Pad to exactly 1024 tokens
                # Using 50256 (GPT-2's EOS token) for padding
                padded_seq = seq + [50256] * (max_length - len(seq))

                x.append(padded_seq)

            # Store label and context length (constant for all 4 endings)
            y.append([example["label"], context_len])

        # Convert to tensors
        x = torch.tensor(x, dtype=torch.long)  # Shape: (batch_size * 4, 1024)
        y = torch.tensor(y, dtype=torch.long)  # Shape: (batch_size, 2)

        return x, y

    def calculate_correctness(self, logits, y, x):
        """
        Calculate the correctness of the predictions using HellaSwag evaluation.

        For each example with 4 endings:
        1. Compute the log probability of each ending's tokens
        2. Sum log probs for each ending (equivalent to multiplying probabilities)
        3. Pick the ending with highest total log probability
        4. Compare with ground truth label

        Args:
            logits: (B*4, 1024, vocab_size) - Model predictions
            y: (B, 2) - Each row is [label, context_length]
            x: (B*4, 1024) - Input token sequences

        Returns:
            num_correct: Number of correct predictions
        """
        num_correct = 0
        B = len(y)  # Number of examples

        for i in range(B):
            # Get the 4 full sequences (context + ending) for this example
            seq_logits = logits[i * 4 : (i + 1) * 4]  # Shape: (4, 1024, vocab_size)
            seq_tokens = x[i * 4 : (i + 1) * 4]  # Shape: (4, 1024)

            # Context length for this example (where endings start)
            context_len = y[i][1].item()

            # Compute log probability for each of the 4 endings
            ending_log_probs = []

            for j in range(4):
                # Extract logits and tokens for just the ending portion (after context)
                ending_logits = seq_logits[
                    j, context_len - 1 : -1
                ]  # Shape: (ending_len, vocab_size)
                ending_tokens = seq_tokens[j, context_len:]  # Shape: (ending_len,)

                # Convert to log probabilities (more numerically stable)
                log_probs = F.log_softmax(
                    ending_logits, dim=-1
                )  # Shape: (ending_len, vocab_size)

                # Get log probabilities for actual tokens using advanced indexing
                # log_probs[i, ending_tokens[i]] for all i in one operation
                token_log_probs = log_probs[
                    torch.arange(len(ending_tokens)), ending_tokens
                ]

                # Filter out padding tokens (50256) and sum log probs
                mask = ending_tokens != 50256
                valid_log_probs = token_log_probs[mask]
                total_log_prob = valid_log_probs.sum().item()

                ending_log_probs.append(total_log_prob)

            # Get the index of the max log probability
            pred_ending = torch.tensor(ending_log_probs).argmax().item()

            # Check if prediction matches ground truth
            if pred_ending == y[i][0].item():
                num_correct += 1

        return num_correct
