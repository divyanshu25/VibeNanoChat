"""
Pre-compute token byte lengths for BPB calculation.
Run once to generate token_bytes.pt
"""

import os
import sys
import torch

# Add parent directory to path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(os.path.join(parent_dir, "src"))

from gpt_2.utils import get_custom_tokenizer, get_special_tokens


def main():
    print("Generating token_bytes.pt...")

    enc, _ = get_custom_tokenizer()
    special_tokens = get_special_tokens()
    special_token_ids = set(special_tokens.values())
    special_token_ids.add(50256)  # GPT-2's <|endoftext|>

    vocab_size = 50262  # GPT-2 base (50257) + 5 special tokens

    token_bytes = []
    for token_id in range(vocab_size):
        if token_id in special_token_ids:
            token_bytes.append(0)  # special tokens don't count
        else:
            try:
                byte_len = len(enc.decode_single_token_bytes(token_id))
                token_bytes.append(byte_len)
            except KeyError:
                token_bytes.append(0)  # unknown tokens don't count

    token_bytes = torch.tensor(token_bytes, dtype=torch.int32)

    output_path = os.path.join(current_dir, "token_bytes.pt")
    torch.save(token_bytes, output_path)

    print(f"Saved to {output_path}")
    print(f"Shape: {token_bytes.shape}")
    print(f"Non-zero tokens: {(token_bytes > 0).sum().item()}")
    print(f"Sample: token_bytes[:10] = {token_bytes[:10].tolist()}")


if __name__ == "__main__":
    main()
