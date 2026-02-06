#!/usr/bin/env python3
"""
Simple Interactive Text Generation Script

This script loads a pretrained GPT model and lets you generate text interactively.
You can type a prompt, adjust temperature and repetition penalty, and see the output.

Usage:
    python debug_tools/interactive_generate.py /path/to/checkpoint.pt
"""

import argparse
import os
import sys

import torch

# Add the src directory to the Python path so we can import our modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from eval_tasks.training import generate
# Import our model and generation utilities
from gpt_2.gpt2_model import GPT
from gpt_2.utils import load_checkpoint


def load_model(checkpoint_path, device):
    """
    Load a pretrained model from a checkpoint file.

    Args:
        checkpoint_path: Path to the .pt checkpoint file
        device: Device to load model on ('cuda', 'mps', or 'cpu')

    Returns:
        Loaded GPT model ready for generation
    """
    print("\n" + "=" * 70)
    print("🔄 Loading model from checkpoint...")
    print("=" * 70)

    # Load the checkpoint file (contains model weights + config)
    # weights_only=False allows loading the config object
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    # Extract the model configuration (defines architecture: layers, embed dim, etc.)
    config = checkpoint["config"]

    # Create a new GPT model instance with this config
    model = GPT(config)

    # Move model to the target device (GPU/CPU)
    model = model.to(device)

    # Set model to evaluation mode (disables dropout, etc.)
    model.eval()

    # Load the trained weights into the model
    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        optimizer=None,  # We don't need optimizer for inference
        master_process=True,
    )

    # Print model info
    total_params = sum(p.numel() for p in model.parameters())
    print("✅ Model loaded successfully!")
    print(f"   Parameters: {total_params:,} ({total_params/1e6:.1f}M)")
    print(f"   Device: {device}")
    print(f"   Layers: {config.n_layer}")
    print(f"   Embedding dimension: {config.n_embed}")
    print(f"   Attention heads: {config.n_head}")
    print(f"   Max sequence length: {config.block_size}")
    print("=" * 70 + "\n")

    return model


def interactive_loop(model, device):
    """
    Run an interactive loop where user can type prompts and get completions.

    Args:
        model: The loaded GPT model
        device: Device the model is on
    """
    print("=" * 70)
    print("🎯 Interactive Text Generation")
    print("=" * 70)
    print("Type your prompt and press Enter to generate text.")
    print("Type 'settings' to adjust temperature and repetition penalty.")
    print("Type 'quit' or 'exit' to stop.")
    print("=" * 70 + "\n")

    # Default generation settings
    temperature = 0.8  # Higher = more random/creative (0.1-2.0 typical range)
    repetition_penalty = (
        1.2  # Higher = less repetition (1.0 = no penalty, 1.5 = strong penalty)
    )
    max_length = 100  # Maximum tokens to generate

    while True:
        # Display current settings
        print("\n📊 Current settings:")
        print(f"   Temperature: {temperature}")
        print(f"   Repetition penalty: {repetition_penalty}")
        print(f"   Max length: {max_length} tokens")

        # Get user input
        print("\n💬 Enter your prompt (or 'settings'/'quit'):")
        user_input = input("> ").strip()

        # Check for special commands
        if user_input.lower() in ["quit", "exit"]:
            print("\n👋 Goodbye!")
            break

        if user_input.lower() == "settings":
            # Let user adjust generation parameters
            print("\n⚙️  Adjust Settings")
            print("-" * 70)

            try:
                # Temperature controls randomness in token selection
                # Low (0.1-0.5): More focused, deterministic
                # Medium (0.6-1.0): Balanced creativity
                # High (1.1-2.0): More random and creative
                temp_input = input(f"Temperature [{temperature}]: ").strip()
                if temp_input:
                    temperature = float(temp_input)
                    temperature = max(
                        0.01, min(2.0, temperature)
                    )  # Clamp to reasonable range

                # Repetition penalty discourages repeating the same tokens
                # 1.0 = no penalty (allow repetition)
                # 1.2 = mild penalty (good default)
                # 1.5+ = strong penalty (very diverse text, but may be incoherent)
                rep_input = input(
                    f"Repetition penalty [{repetition_penalty}]: "
                ).strip()
                if rep_input:
                    repetition_penalty = float(rep_input)
                    repetition_penalty = max(1.0, min(2.0, repetition_penalty))

                # Max length is the total number of tokens (prompt + generated)
                len_input = input(f"Max length [{max_length}]: ").strip()
                if len_input:
                    max_length = int(len_input)
                    max_length = max(10, min(2048, max_length))  # Reasonable bounds

                print("✅ Settings updated!")

            except ValueError:
                print("❌ Invalid input! Settings unchanged.")

            continue

        # Skip empty prompts
        if not user_input:
            print("❌ Please enter a prompt!")
            continue

        # Generate text from the prompt
        print("\n" + "=" * 70)
        print("🚀 Generating...")
        print("=" * 70)

        try:
            # Prepend BOS token to match training data format
            # The model was trained with BOS-aligned sequences, so we should include it
            # Note: User can still manually include BOS by typing <|bos|> in their prompt
            if not user_input.startswith("<|bos|>"):
                context_with_bos = "<|bos|>" + user_input
            else:
                context_with_bos = user_input

            # Create a random number generator for sampling
            # Using a fixed seed (42) makes generation deterministic for the same prompt
            rng = torch.Generator(device=device)
            rng.manual_seed(42)

            # Call the generate function (uses KV cache for speed!)
            # This function:
            # 1. Tokenizes the input prompt (including BOS token)
            # 2. Runs model forward pass to get logits (predictions)
            # 3. Samples tokens one at a time using temperature and top-k
            # 4. Applies repetition penalty to avoid redundancy
            # 5. Decodes tokens back to text
            outputs = generate(
                num_sequences=1,  # Generate 1 completion
                max_length=max_length,  # Total tokens (prompt + generated)
                model=model,  # Our loaded model
                context=context_with_bos,  # The user's prompt with BOS prepended
                device=device,  # GPU/CPU
                random_number_generator=rng,  # For sampling randomness
                use_kv_cache=True,  # Enable KV cache (much faster!)
                verbose=False,  # Don't print progress updates
                temperature=temperature,  # Randomness level
                top_k=50,  # Only sample from top 50 tokens
                repetition_penalty=repetition_penalty,  # Discourage repetition
            )

            # Display the generated text
            print("\n" + "=" * 70)
            print("✨ Generated Text:")
            print("=" * 70)
            print(outputs[0])  # outputs is a list, get the first (only) result
            print("=" * 70)

        except Exception as e:
            # If something goes wrong, show the error
            print(f"\n❌ Error during generation: {e}")
            import traceback

            traceback.print_exc()

            # Clean up any GPU memory from the failed generation
            if device == "cuda":
                torch.cuda.empty_cache()
                print("🧹 Cleared GPU cache after error")


def main():
    """
    Main entry point for the script.
    Parses command line arguments and starts the interactive loop.
    """
    # Set up command line argument parsing
    parser = argparse.ArgumentParser(
        description="Interactive text generation with a pretrained GPT model"
    )
    parser.add_argument(
        "checkpoint_path", type=str, help="Path to the model checkpoint (.pt file)"
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        choices=["cuda", "mps", "cpu"],
        help="Device to run on (default: auto-detect)",
    )

    args = parser.parse_args()

    # Auto-detect device if not specified
    if args.device is None:
        if torch.cuda.is_available():
            device = "cuda"
        elif torch.backends.mps.is_available():
            device = "mps"
        else:
            device = "cpu"
    else:
        device = args.device

    # Check if checkpoint exists
    if not os.path.exists(args.checkpoint_path):
        print(f"❌ Error: Checkpoint file not found: {args.checkpoint_path}")
        return 1

    # Load the model
    try:
        model = load_model(args.checkpoint_path, device)
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        import traceback

        traceback.print_exc()
        return 1

    # Start interactive loop with proper cleanup
    try:
        interactive_loop(model, device)
    except KeyboardInterrupt:
        print("\n\n👋 Interrupted by user. Goodbye!")
    finally:
        # ===== CRITICAL: Clean up GPU resources =====
        # This ensures GPU memory is released when the script exits
        # Without this, the model stays in GPU memory until Python process ends
        print("\n🧹 Cleaning up GPU resources...")

        # Delete model to free references
        del model

        # Force garbage collection to clean up Python objects
        import gc

        gc.collect()

        # Clear CUDA cache (if using CUDA)
        if device == "cuda":
            torch.cuda.empty_cache()
            # Synchronize to ensure all operations complete
            torch.cuda.synchronize()

        print("✅ GPU cleanup complete!")

    return 0


# This runs when you execute the script directly
if __name__ == "__main__":
    exit(main())
