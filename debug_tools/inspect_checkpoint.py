#!/usr/bin/env python3
"""
Simple script to inspect a checkpoint and show vocabulary size.
"""

import sys

import torch


def inspect_checkpoint(checkpoint_path):
    """Load and inspect a model checkpoint."""
    print(f"\n{'='*80}")
    print(f"Inspecting checkpoint: {checkpoint_path}")
    print(f"{'='*80}\n")

    try:
        # Load checkpoint
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

        # Get model state dict
        model_state = checkpoint.get("model", {})

        # Check for embedding weights
        if "transformer.wte.weight" in model_state:
            vocab_size = model_state["transformer.wte.weight"].shape[0]
            embed_dim = model_state["transformer.wte.weight"].shape[1]

            print("✅ Token Embeddings (transformer.wte.weight):")
            print(f"   Vocabulary size: {vocab_size}")
            print(f"   Embedding dimension: {embed_dim}")
            print()

        # Check for LM head weights
        if "lm_head.weight" in model_state:
            lm_vocab_size = model_state["lm_head.weight"].shape[0]
            lm_embed_dim = model_state["lm_head.weight"].shape[1]

            print("✅ LM Head (lm_head.weight):")
            print(f"   Vocabulary size: {lm_vocab_size}")
            print(f"   Embedding dimension: {lm_embed_dim}")
            print()

        # Check if weights are tied
        if "transformer.wte.weight" in model_state and "lm_head.weight" in model_state:
            are_tied = torch.equal(
                model_state["transformer.wte.weight"], model_state["lm_head.weight"]
            )
            print(f"Weights tied: {'Yes' if are_tied else 'No'}")
            print()

        # Show other checkpoint metadata
        print("Checkpoint metadata:")
        if "epoch" in checkpoint:
            print(f"   Epoch: {checkpoint['epoch']}")
        if "step" in checkpoint:
            print(f"   Step: {checkpoint['step']}")
        if "global_step" in checkpoint:
            print(f"   Global step: {checkpoint['global_step']}")
        if "val_loss" in checkpoint:
            print(f"   Validation loss: {checkpoint['val_loss']:.4f}")

        # Show config if available
        if "config" in checkpoint:
            config = checkpoint["config"]
            print("\nModel configuration:")
            print(f"   vocab_size: {config.vocab_size}")
            print(f"   block_size: {config.block_size}")
            print(f"   n_layer: {config.n_layer}")
            print(f"   n_head: {config.n_head}")
            print(f"   n_embed: {config.n_embed}")

        print(f"\n{'='*80}\n")

    except Exception as e:
        print(f"❌ Error loading checkpoint: {e}")
        sys.exit(1)


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python inspect_checkpoint.py <checkpoint_path>")
        sys.exit(1)

    checkpoint_path = sys.argv[1]
    inspect_checkpoint(checkpoint_path)


# uv run python inspect_checkpoint.py /sensei-fs/users/divgoyal/nanogpt/pretrain_checkpoints/model_checkpoint_global37953_pretraining.pt
