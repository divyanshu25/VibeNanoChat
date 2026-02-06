#!/usr/bin/env python3
"""
Simple Chat Script for VibeNanoChat

Load a checkpoint and chat with the model interactively.

Usage:
    python scripts/chat.py --checkpoint /path/to/checkpoint.pt
"""

import argparse
import os
import sys

# Add gpt_2 to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.append(src_dir)

import torch
import torch.nn.functional as F

from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT
from gpt_2.utils import get_custom_tokenizer, load_checkpoint


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def generate_response(model, tokenizer, prompt, device, max_new_tokens=256):
    """Generate a response from the model given a prompt."""
    model.eval()

    tokens = tokenizer.encode(prompt, allowed_special="all")
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    prompt_length = tokens.size(1)

    stop_tokens = [
        tokenizer.encode("<|assistant_end|>", allowed_special="all")[0],
        tokenizer.encode("<|endoftext|>", allowed_special="all")[0],
        tokenizer.encode("<|user_start|>", allowed_special="all")[0],
        tokenizer.encode("<|user_end|>", allowed_special="all")[0],
    ]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if tokens.size(1) > model.config.block_size:
                tokens = tokens[:, -model.config.block_size :]

            logits, _ = model(tokens)
            logits = logits[:, -1, :] / 0.8  # temperature

            # Top-k sampling
            v, _ = torch.topk(logits, 50)
            logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() in stop_tokens:
                break

    generated_tokens = tokens[0, prompt_length:].tolist()
    raw_response = tokenizer.decode(generated_tokens)

    # Highlight special tokens in red brackets for debugging
    RED = "\033[91m"
    RESET = "\033[0m"
    response = raw_response
    for token_str in [
        "<|assistant_end|>",
        "<|endoftext|>",
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
        "<|bos|>",
    ]:
        response = response.replace(token_str, f"{RED}[{token_str}]{RESET}")

    return response.strip(), raw_response


def format_chat_prompt(messages):
    """Format conversation history into the chat prompt format."""
    prompt = "<|bos|>"
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"<|user_start|>{msg['content']}<|user_end|>"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant_start|>{msg['content']}<|assistant_end|>"
    prompt += "<|assistant_start|>"
    return prompt


def main():
    parser = argparse.ArgumentParser(description="Chat with a VibeNanoChat checkpoint")
    parser.add_argument(
        "--checkpoint", "-c", type=str, required=True, help="Path to checkpoint (.pt)"
    )
    args = parser.parse_args()

    if not os.path.exists(args.checkpoint):
        print(f"❌ Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    device = get_device()
    print(f"🖥️  Device: {device}")

    # Load model
    print("🔧 Loading model...")
    tokenizer, _ = get_custom_tokenizer()
    model = GPT(GPTConfig())
    model.to(device)

    load_checkpoint(args.checkpoint, model, device, optimizer=None, master_process=True)
    model.eval()

    print("\n" + "=" * 50)
    print("🤖 VibeNanoChat")
    print("Type /quit to exit, /clear to reset conversation")
    print("=" * 50 + "\n")

    messages = []

    while True:
        try:
            user_input = input("You: ").strip()

            if not user_input:
                continue

            if user_input.lower() in ["/quit", "/exit", "/q"]:
                print("👋 Bye!")
                break

            if user_input.lower() == "/clear":
                messages = []
                print("🗑️  Cleared.\n")
                continue

            messages.append({"role": "user", "content": user_input})
            prompt = format_chat_prompt(messages)

            # Debug: show prompt with special tokens
            print(f"\n[INPUT] {prompt}")

            response, raw_response = generate_response(model, tokenizer, prompt, device)

            # Debug: show raw response with special tokens
            print(f"[OUTPUT] {raw_response}")

            print(f"\nAssistant: {response}\n")

            messages.append({"role": "assistant", "content": response})

        except KeyboardInterrupt:
            print("\n👋 Bye!")
            break


if __name__ == "__main__":
    main()
