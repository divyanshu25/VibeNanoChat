#!/usr/bin/env python3
"""
ChatCORE Integration Test with KV Caching

⚠️  NOTE: This is an INTEGRATION test, not a correctness test!
    For KV cache correctness verification, use: test_kvcache_correctness.py

This script tests the end-to-end integration of:
- Model checkpoint loading
- ChatCORE evaluator setup
- GSM8K evaluation pipeline
- KV caching during generation
- Tool use (calculator) integration

What this DOES test:
✓ System integration works without crashes
✓ Evaluation pipeline completes successfully
✓ Performance is reasonable with KV cache

What this DOES NOT test:
✗ KV cache produces correct outputs (no comparison with non-cached)
✗ Edge cases (empty prompts, single tokens, etc.)
✗ Deterministic correctness verification

Usage:
    # Run with KV cache enabled (default - faster)
    python test_chatcore_kvcache.py

    # Run without KV cache (slower, for comparison)
    python test_chatcore_kvcache.py --no-kv-cache

    # Customize number of examples and max tokens
    python test_chatcore_kvcache.py --max-examples 10 --max-tokens 256

Output:
    - GSM8K accuracy on test examples
    - Generation speed statistics with/without KV cache
    - Performance metrics and throughput
"""

import argparse
import os
import sys
import time

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from eval_tasks.chat_core.evaluator import ChatCoreEvaluator
from eval_tasks.chat_core.utils import setup_gsm8k_task
from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT
from gpt_2.utils import get_custom_tokenizer, load_checkpoint


def main():
    # =========================================================================
    # ARGUMENT PARSING
    # =========================================================================
    parser = argparse.ArgumentParser(
        description="Test ChatCORE evaluation with optional KV caching"
    )
    parser.add_argument(
        "--no-kv-cache",
        action="store_true",
        help="Disable KV caching (slower but useful for comparison)",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=20,
        help="Number of examples to evaluate (default: 20)",
    )
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=512,
        help="Maximum tokens to generate (default: 512)",
    )
    args = parser.parse_args()

    use_kv_cache = not args.no_kv_cache

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    checkpoint_path = "/sensei-fs/users/divgoyal/nanogpt/midtrain_checkpoints/model_checkpoint_global2633_midtraining.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Evaluation settings
    max_examples = args.max_examples
    max_tokens = args.max_tokens
    temperature = 0.0  # Greedy decoding for consistency

    print("\n" + "=" * 80)
    print("🧪 ChatCORE KV Cache Test Script")
    print("=" * 80)
    print(f"Checkpoint: {checkpoint_path}")
    print(f"Device: {device}")
    print(f"Max examples: {max_examples}")
    print(f"Max tokens per generation: {max_tokens}")
    print(f"Temperature: {temperature}")
    print(f"KV Cache: {'✅ ENABLED' if use_kv_cache else '❌ DISABLED'}")
    print("=" * 80 + "\n")

    # =========================================================================
    # STEP 1: Load checkpoint to get config
    # =========================================================================
    print("📂 Loading checkpoint to extract config...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)

    if "config" in checkpoint:
        config = checkpoint["config"]
        print("✅ Config loaded from checkpoint")
    else:
        # Fallback: create default config
        print("⚠️  No config in checkpoint, using default")
        config = GPTConfig(
            n_layer=12,
            n_head=12,
            n_embed=768,
            block_size=1024,
            vocab_size=50304,
        )

    print(
        f"   Model config: {config.n_layer} layers, {config.n_head} heads, "
        f"{config.n_embed} embed dim, vocab={config.vocab_size}"
    )

    # =========================================================================
    # STEP 2: Load tokenizer
    # =========================================================================
    print("\n📝 Loading tokenizer...")
    try:
        tokenizer, special_tokens = get_custom_tokenizer()
        print(f"✅ Tokenizer loaded (vocab size: {tokenizer.n_vocab})")

        # Verify tokenizer has tool tokens
        has_tool_tokens = "<|python|>" in special_tokens
        print(f"   Tool tokens available: {has_tool_tokens}")
    except Exception as e:
        print(f"❌ Failed to load tokenizer: {e}")
        import traceback

        traceback.print_exc()
        return

    # =========================================================================
    # STEP 3: Initialize model
    # =========================================================================
    print("\n🏗️  Initializing model...")
    model = GPT(config)
    model = model.to(device)
    print(f"✅ Model initialized and moved to {device}")

    # Load weights from checkpoint
    print("\n📥 Loading model weights from checkpoint...")
    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        optimizer=None,
        master_process=True,
    )
    print("✅ Model weights loaded successfully")

    # Get model info
    total_params = sum(p.numel() for p in model.parameters())
    print(f"   Total parameters: {total_params:,} ({total_params/1e6:.1f}M)")

    # =========================================================================
    # STEP 4: Set up ChatCORE evaluator
    # =========================================================================
    print("\n🎯 Setting up ChatCORE evaluator...")
    model.eval()  # Put model in eval mode

    evaluator = ChatCoreEvaluator(
        model=model,
        tokenizer=tokenizer,
        device=device,
        master_process=True,
        ddp=False,
        ddp_rank=0,
        ddp_world_size=1,
        max_examples=max_examples,
        num_samples=1,
        max_tokens=max_tokens,
        temperature=temperature,
        top_k=50,
        use_kv_cache=use_kv_cache,
    )
    print("✅ ChatCORE evaluator initialized")

    # =========================================================================
    # STEP 5: Register GSM8K task
    # =========================================================================
    print("\n📋 Registering GSM8K task...")
    try:
        setup_gsm8k_task(
            evaluator=evaluator,
            tokenizer=tokenizer,
            split="test",
            cache_dir="/sensei-fs/users/divgoyal/nanochat_midtraining_data",
        )
        print("✅ GSM8K task registered")
    except Exception as e:
        print(f"❌ Failed to register GSM8K: {e}")
        return

    # =========================================================================
    # STEP 6: Run evaluation 3 times and compute averages
    # =========================================================================
    print("\n" + "=" * 80)
    if use_kv_cache:
        print("🚀 Running ChatCORE evaluation with KV caching (3 runs)...")
    else:
        print("🚀 Running ChatCORE evaluation without KV caching (3 runs)...")
    print("=" * 80 + "\n")

    num_runs = 3
    all_run_results = []
    all_run_times = []

    try:
        for run_idx in range(num_runs):
            print(f"\n{'─'*80}")
            print(f"Run {run_idx + 1}/{num_runs}")
            print(f"{'─'*80}")

            start_time = time.time()
            results = evaluator.evaluate_all_tasks(step=0, global_step=2633)
            elapsed_time = time.time() - start_time

            all_run_results.append(results)
            all_run_times.append(elapsed_time)

            # Display results for this run
            for task_name, task_results in results.items():
                accuracy = task_results["accuracy"]
                correct = task_results["correct"]
                total = task_results["total"]

                print(f"\n{task_name}:")
                print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
                print(f"  Correct:  {correct}/{total}")

            print(f"\nRun time: {elapsed_time:.2f}s")
            print(f"Average time per example: {elapsed_time/max_examples:.2f}s")

        # =====================================================================
        # STEP 7: Compute and display average results across all runs
        # =====================================================================
        print("\n" + "=" * 80)
        print("📊 AVERAGE RESULTS ACROSS 3 RUNS")
        print("=" * 80)

        # Calculate average accuracy per task
        task_names = list(all_run_results[0].keys())
        for task_name in task_names:
            accuracies = [
                run_results[task_name]["accuracy"] for run_results in all_run_results
            ]
            corrects = [
                run_results[task_name]["correct"] for run_results in all_run_results
            ]
            totals = [
                run_results[task_name]["total"] for run_results in all_run_results
            ]

            avg_accuracy = sum(accuracies) / len(accuracies)
            avg_correct = sum(corrects) / len(corrects)
            avg_total = sum(totals) / len(totals)

            # Calculate standard deviation
            std_accuracy = (
                sum((acc - avg_accuracy) ** 2 for acc in accuracies) / len(accuracies)
            ) ** 0.5

            print(f"\n{task_name}:")
            print(f"  Average Accuracy: {avg_accuracy:.4f} ({avg_accuracy*100:.2f}%)")
            print(f"  Std Dev:          {std_accuracy:.4f} ({std_accuracy*100:.2f}%)")
            print(f"  Average Correct:  {avg_correct:.1f}/{avg_total:.1f}")
            print(f"  Individual runs:  {[f'{acc:.4f}' for acc in accuracies]}")

        # Calculate average timing
        avg_time = sum(all_run_times) / len(all_run_times)
        std_time = (
            sum((t - avg_time) ** 2 for t in all_run_times) / len(all_run_times)
        ) ** 0.5

        print("\nTiming:")
        print(f"  Average total time:       {avg_time:.2f}s")
        print(f"  Std Dev:                  {std_time:.2f}s")
        print(f"  Average time per example: {avg_time/max_examples:.2f}s")
        print(f"  Individual runs:          {[f'{t:.2f}s' for t in all_run_times]}")
        print("=" * 80)

        # =====================================================================
        # STEP 8: Integration status report
        # =====================================================================
        print("\n" + "=" * 80)
        print("✅ INTEGRATION STATUS")
        print("=" * 80)
        if use_kv_cache:
            print("✓ Evaluation completed successfully with KV caching")
        else:
            print("✓ Evaluation completed successfully without KV caching")
        print("✓ No crashes or errors during generation")
        print("✓ Pipeline integration appears functional")

        print("\n⚠️  NOTE: This test does NOT verify correctness!")
        print("   To verify KV cache produces correct outputs:")
        print("   → Run: python debug_tools/test_kvcache_correctness.py")

        if avg_time / max_examples < 10:
            if use_kv_cache:
                print("\n✓ Generation speed looks reasonable (KV cache likely helping)")
            else:
                print("\n✓ Generation speed looks good even without KV cache")
        else:
            print("\n⚠️  Generation seems slow - verify GPU is being used")

        print("=" * 80 + "\n")

    except Exception as e:
        print("\n❌ Evaluation failed with error:")
        print(f"   {type(e).__name__}: {e}")
        import traceback

        traceback.print_exc()
        return

    print("\n✅ Test completed successfully!\n")


if __name__ == "__main__":
    main()
