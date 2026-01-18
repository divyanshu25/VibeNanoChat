#!/usr/bin/env python3
"""
Test Script for Generate Function with KV Caching

This script validates the generate function's KV cache implementation:
1. Correctness: KV cache produces identical outputs to non-cached
2. Speed: KV cache is significantly faster
3. Prefill optimization: Batch generation uses efficient prefill pattern

Usage:
    python debug_tools/test_generate_kvcache.py
"""

import os
import sys
import time

import torch

# Add src to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

from eval_tasks.training import generate
from gpt_2.gpt2_model import GPT
from gpt_2.utils import get_custom_tokenizer, load_checkpoint


def test_correctness(model, device, tokenizer):
    """
    Test that KV cache produces identical results to non-cached generation.
    """
    print("\n" + "=" * 80)
    print("TEST 1: Correctness - KV Cache vs No Cache")
    print("=" * 80)

    context = "Once upon a time"
    num_sequences = 1
    max_length = 50
    seed = 42

    print(f"Context: '{context}'")
    print(f"Generating {max_length} tokens with seed={seed}")

    # Create RNG
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # Generate WITHOUT cache
    print("\n🔄 Generating WITHOUT KV cache...")
    start = time.time()
    result_no_cache = generate(
        num_sequences=num_sequences,
        max_length=max_length,
        model=model,
        context=context,
        device=device,
        random_number_generator=rng,
        use_kv_cache=False,
    )
    time_no_cache = time.time() - start

    # Reset RNG to same seed
    rng = torch.Generator(device=device)
    rng.manual_seed(seed)

    # Generate WITH cache
    print("🚀 Generating WITH KV cache...")
    start = time.time()
    result_with_cache = generate(
        num_sequences=num_sequences,
        max_length=max_length,
        model=model,
        context=context,
        device=device,
        random_number_generator=rng,
        use_kv_cache=True,
    )
    time_with_cache = time.time() - start

    # Compare results
    print("\n" + "-" * 80)
    print("RESULTS:")
    print("-" * 80)

    if result_no_cache[0] == result_with_cache[0]:
        print("✅ PASS: Both methods generated IDENTICAL text!")
        print("\n📊 Performance:")
        print(f"   Without cache: {time_no_cache:.3f}s")
        print(f"   With cache:    {time_with_cache:.3f}s")
        print(f"   Speedup:       {time_no_cache/time_with_cache:.2f}x")

        # Show first 200 chars of output
        output_preview = result_with_cache[0][:200]
        if len(result_with_cache[0]) > 200:
            output_preview += "..."
        print(f"\n📝 Generated text:\n{output_preview}")
        return True
    else:
        print("❌ FAIL: Generated text differs!")
        print(f"\nWithout cache:\n{result_no_cache[0][:200]}...")
        print(f"\nWith cache:\n{result_with_cache[0][:200]}...")

        # Find first difference
        tokens_no_cache = tokenizer.encode(result_no_cache[0])
        tokens_with_cache = tokenizer.encode(result_with_cache[0])
        for i, (t1, t2) in enumerate(zip(tokens_no_cache, tokens_with_cache)):
            if t1 != t2:
                print(f"\n⚠️  First difference at token position {i}")
                print(f"   Without cache: {t1} -> '{tokenizer.decode([t1])}'")
                print(f"   With cache: {t2} -> '{tokenizer.decode([t2])}'")
                break
        return False


def test_batch_generation(model, device):
    """
    Test batch generation with prefill optimization.
    """
    print("\n" + "=" * 80)
    print("TEST 2: Batch Generation with Prefill Optimization")
    print("=" * 80)

    context = "The quick brown fox"
    max_length = 100
    seed = 123

    print(f"Context: '{context}'")
    print("Testing batch sizes: [1, 2, 4, 8]")

    results = []

    for batch_size in [1, 2, 4, 8]:
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)

        print(f"\n📦 Batch size: {batch_size}")

        start = time.time()
        outputs = generate(
            num_sequences=batch_size,
            max_length=max_length,
            model=model,
            context=context,
            device=device,
            random_number_generator=rng,
            use_kv_cache=True,
        )
        elapsed = time.time() - start

        tokens_per_seq = max_length - len(context.split())
        total_tokens = tokens_per_seq * batch_size
        throughput = total_tokens / elapsed

        print(f"   Time: {elapsed:.3f}s")
        print(f"   Throughput: {throughput:.1f} tokens/sec")

        results.append(
            {"batch_size": batch_size, "time": elapsed, "throughput": throughput}
        )

        # Show diversity (first 80 chars of each)
        if batch_size <= 4:
            print(f"   Generated {len(outputs)} diverse sequences:")
            for i, out in enumerate(outputs, 1):
                preview = out[:80]
                if len(out) > 80:
                    preview += "..."
                print(f"   {i}. {preview}")

    # Analyze scaling
    print("\n" + "-" * 80)
    print("PREFILL OPTIMIZATION ANALYSIS:")
    print("-" * 80)
    print(
        f"{'Batch Size':>12} | {'Time (s)':>10} | {'Tokens/sec':>12} | {'Efficiency':>12}"
    )
    print("-" * 60)

    baseline_time = results[0]["time"]
    for r in results:
        efficiency = (baseline_time * r["batch_size"]) / r["time"]
        print(
            f"{r['batch_size']:>12} | {r['time']:>10.3f} | {r['throughput']:>12.1f} | {efficiency:>11.1f}x"
        )

    print("\n💡 Efficiency interpretation:")
    print("   - 1.0x = Linear scaling (ideal)")
    print("   - >0.8x = Good efficiency (prefill optimization working)")
    print("   - <0.5x = Poor efficiency (possible bottleneck)")

    # Check if prefill is helping
    efficiency_score = (baseline_time * results[-1]["batch_size"]) / results[-1]["time"]
    if efficiency_score > 0.8:
        print(f"\n✅ PASS: Good batch efficiency ({efficiency_score:.2f}x)")
        print("   Prefill optimization is working effectively!")
        return True
    else:
        print(f"\n⚠️  WARNING: Low batch efficiency ({efficiency_score:.2f}x)")
        print("   Expected >0.8x for effective prefill optimization")
        return False


def test_long_sequence_speedup(model, device):
    """
    Test speedup on longer sequences where KV cache benefit is most visible.
    """
    print("\n" + "=" * 80)
    print("TEST 3: Long Sequence Generation Speedup")
    print("=" * 80)

    context = "In a world where"
    lengths = [50, 100, 200, 400]
    seed = 456

    print(f"Context: '{context}'")
    print(f"Testing sequence lengths: {lengths}")

    print(f"\n{'Length':>8} | {'No Cache':>12} | {'With Cache':>12} | {'Speedup':>10}")
    print("-" * 50)

    speedups = []

    for max_length in lengths:
        # Without cache
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        start = time.time()
        _ = generate(
            num_sequences=1,
            max_length=max_length,
            model=model,
            context=context,
            device=device,
            random_number_generator=rng,
            use_kv_cache=False,
        )
        time_no_cache = time.time() - start

        # With cache
        rng = torch.Generator(device=device)
        rng.manual_seed(seed)
        start = time.time()
        _ = generate(
            num_sequences=1,
            max_length=max_length,
            model=model,
            context=context,
            device=device,
            random_number_generator=rng,
            use_kv_cache=True,
        )
        time_with_cache = time.time() - start

        speedup = time_no_cache / time_with_cache
        speedups.append(speedup)

        print(
            f"{max_length:>8} | {time_no_cache:>10.3f}s | {time_with_cache:>10.3f}s | {speedup:>9.2f}x"
        )

    # Check if speedup increases with length
    print("\n" + "-" * 80)
    if speedups[-1] > speedups[0]:
        print("✅ PASS: Speedup increases with sequence length")
        print(
            f"   {lengths[0]} tokens: {speedups[0]:.2f}x → {lengths[-1]} tokens: {speedups[-1]:.2f}x"
        )
        print("   This confirms O(N²) → O(N) complexity reduction!")
        return True
    else:
        print("⚠️  WARNING: Speedup doesn't scale as expected")
        print(
            f"   {lengths[0]} tokens: {speedups[0]:.2f}x → {lengths[-1]} tokens: {speedups[-1]:.2f}x"
        )
        return False


def main():
    """
    Run all tests for the generate function with KV caching.
    """
    print("\n" + "=" * 80)
    print("🧪 TESTING GENERATE FUNCTION WITH KV CACHING")
    print("=" * 80)

    # =========================================================================
    # Setup
    # =========================================================================
    checkpoint_path = "/sensei-fs/users/divgoyal/nanogpt/pretrain_checkpoints/model_checkpoint_global37953_pretraining.pt"
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"\n📂 Checkpoint: {os.path.basename(checkpoint_path)}")
    print(f"🖥️  Device: {device}")

    # Load model
    print("\n🔄 Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = GPT(config)
    model = model.to(device)
    model.eval()

    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        optimizer=None,
        master_process=True,
    )

    total_params = sum(p.numel() for p in model.parameters())
    print(f"✅ Model loaded: {total_params:,} parameters ({total_params/1e6:.1f}M)")

    # Load tokenizer
    tokenizer, _ = get_custom_tokenizer()

    # =========================================================================
    # Run Tests
    # =========================================================================
    test_results = []

    # Test 1: Correctness
    try:
        result = test_correctness(model, device, tokenizer)
        test_results.append(("Correctness", result))
    except Exception as e:
        print(f"\n❌ Test 1 failed with error: {e}")
        import traceback

        traceback.print_exc()
        test_results.append(("Correctness", False))

    # Test 2: Batch generation with prefill
    try:
        result = test_batch_generation(model, device)
        test_results.append(("Batch Prefill", result))
    except Exception as e:
        print(f"\n❌ Test 2 failed with error: {e}")
        import traceback

        traceback.print_exc()
        test_results.append(("Batch Prefill", False))

    # Test 3: Long sequence speedup
    try:
        result = test_long_sequence_speedup(model, device)
        test_results.append(("Speedup Scaling", result))
    except Exception as e:
        print(f"\n❌ Test 3 failed with error: {e}")
        import traceback

        traceback.print_exc()
        test_results.append(("Speedup Scaling", False))

    # =========================================================================
    # Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("📊 TEST SUMMARY")
    print("=" * 80)

    for test_name, passed in test_results:
        status = "✅ PASS" if passed else "❌ FAIL"
        print(f"{status:>10} | {test_name}")

    all_passed = all(result for _, result in test_results)

    print("\n" + "=" * 80)
    if all_passed:
        print("🎉 ALL TESTS PASSED!")
        print("   ✓ KV cache generates correct outputs")
        print("   ✓ Prefill optimization works for batch generation")
        print("   ✓ Speedup scales with sequence length")
    else:
        print("⚠️  SOME TESTS FAILED")
        print("   Please review the failures above")
    print("=" * 80 + "\n")

    return 0 if all_passed else 1


if __name__ == "__main__":
    exit(main())
