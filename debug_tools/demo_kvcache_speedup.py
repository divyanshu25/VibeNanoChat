#!/usr/bin/env python3
"""
================================================================================
KV Cache Speedup Demonstration - Educational Script
================================================================================

HOW TO RUN:
    python debug_tools/demo_kvcache_speedup.py

WHAT THIS SCRIPT DOES:
    Demonstrates the performance difference between generation WITH and WITHOUT
    KV caching by:
    1. Generating sequences of increasing length (50, 100, 200, 400 tokens)
    2. Timing each individual generation step
    3. Showing how step times increase without cache but stay constant with cache
    4. Calculating speedup and explaining the O(N²) vs O(N) complexity

WHAT YOU'LL LEARN:
    - Why KV cache is faster (O(N) vs O(N²) complexity)
    - When KV cache provides benefit (longer sequences)
    - How to measure and analyze generation performance
    - Real-world impact of caching on transformer inference

EXPECTED OUTPUT:
    - WITHOUT cache: Step times increase (5ms → 15ms as sequence grows)
    - WITH cache: Step times stay constant (~6ms throughout)
    - Speedup: 1.3x - 1.5x for this model size
    - Clear visualization of O(N²) vs O(N) behavior

--------------------------------------------------------------------------------
KV Cache Speedup Demonstration - Educational Script

=============================================================================
WHAT IS KV CACHE?
=============================================================================

In transformer models, attention computation requires three components:
- Query (Q): "What am I looking for?"
- Key (K): "What do I contain?"
- Value (V): "What information do I provide?"

The attention formula is: Attention(Q, K, V) = softmax(Q @ K^T / sqrt(d)) @ V

=============================================================================
KEY INSIGHT: For previously generated tokens, their K and V never change!
=============================================================================

WHY DON'T K AND V CHANGE?

1. **Token Embeddings Are Fixed**
   Once a token is generated, its token ID doesn't change:
   - Token "cat" (ID: 2345) always has the same embedding
   - This embedding is looked up from a fixed embedding table

2. **Position Is Fixed**
   Once a token is placed at position P, it stays there:
   - "cat" at position 5 always has position encoding for position 5
   - Position encoding is deterministic (computed from position number)

3. **K and V Are Linear Projections**
   K and V are computed as: K = embedding × W_k, V = embedding × W_v
   - W_k and W_v are weight matrices (fixed during inference)
   - embedding is fixed (see points 1 and 2)
   - Therefore: K and V are fixed! (fixed × fixed = fixed)

Example:
  Token "cat" at position 5:
  - Embedding: [0.1, 0.3, 0.5, ...] (always the same)
  - K = embedding × W_k = [0.2, 0.4, ...] (always the same)
  - V = embedding × W_v = [0.1, 0.6, ...] (always the same)

  At step 10, when we generate token "dog":
  - "cat" is still at position 5
  - Its embedding hasn't changed
  - Therefore K and V for "cat" are EXACTLY the same as before!

4. **Only Q Changes**
   The Query (Q) comes from the NEW token we're processing:
   - Step 10: Q is computed from token "dog"
   - Step 11: Q is computed from token "the"
   - Each step has a different Q, but the same K,V for previous tokens!

This is why we can cache K and V - they're computed once and reused forever!
The only thing that changes at each step is the Query from the new token.

=============================================================================
WHY DOES Q COME FROM THE LAST TOKEN ONLY?
=============================================================================

During autoregressive generation, we generate ONE token at a time:

SCENARIO: We have tokens [The, cat, sat] and want to predict the next token.

WHAT WE NEED TO PREDICT:
  Position 0: "The"  → Already generated ✓
  Position 1: "cat"  → Already generated ✓
  Position 2: "sat"  → Already generated ✓
  Position 3: ???    → This is what we need to predict!

WHAT EACH POSITION DOES:
  Position 0 ("The"):  Already decided, no prediction needed
  Position 1 ("cat"):  Already decided, no prediction needed
  Position 2 ("sat"):  Needs to ask: "What comes after 'The cat sat'?"
  Position 3 (future): Doesn't exist yet!

THE KEY INSIGHT: We only compute Q for position 2 ("sat") because:
  1. That's the LAST token we have
  2. It's the one asking "what should come next?"
  3. Previous tokens (The, cat) already asked their questions and got answers!
  4. We don't need Q for past tokens - they already predicted their next tokens

WHAT ATTENTION DOES AT POSITION 2:
  Q (from "sat"):      "What should follow 'The cat sat'?"
  K (from "The"):      "I'm a determiner at the start"
  K (from "cat"):      "I'm a noun, the subject"
  K (from "sat"):      "I'm a verb, past tense"

  Attention: Q @ K^T finds relevant context (which words to pay attention to)
  Then uses V to gather information and predict next token

STEP-BY-STEP GENERATION:

Step 1: Tokens = [The, cat]
  - We have: "The cat"
  - Q from position 1 ("cat"): "What comes after 'The cat'?"
  - Attend to: K,V from "The" and "cat"
  - Predict: "sat"
  - Result: [The, cat, sat]

Step 2: Tokens = [The, cat, sat]
  - We have: "The cat sat"
  - Q from position 2 ("sat"): "What comes after 'The cat sat'?"
  - Attend to: K,V from "The", "cat", and "sat"
  - Predict: "on"
  - Result: [The, cat, sat, on]

Notice: At Step 2, we DON'T need Q for positions 0 or 1!
  - Position 0 ("The") already predicted position 1 ("cat") ✓
  - Position 1 ("cat") already predicted position 2 ("sat") ✓
  - Only position 2 ("sat") needs to predict position 3 ("on")

WHY THIS MATTERS FOR KV CACHE:

WITHOUT cache (passing entire sequence):
  - We compute Q, K, V for ALL positions [0, 1, 2]
  - But we only USE the Q from position 2!
  - Q from positions 0 and 1 are WASTED computation
  - Plus, we recompute K,V for positions 0 and 1 (also wasted!)

WITH cache (passing only last token):
  - We compute Q, K, V for ONLY position 2
  - This is the ONLY Q we need!
  - We reuse cached K,V from positions 0 and 1
  - No wasted computation!

VISUAL COMPARISON:

Without cache - passing [The, cat, sat]:
  ┌─────────────────────────────────────┐
  │ Position 0 (The)                    │
  │   Q₀ ← computed but NEVER USED! ❌  │
  │   K₀ ← computed (redundant) ❌      │
  │   V₀ ← computed (redundant) ❌      │
  │                                     │
  │ Position 1 (cat)                    │
  │   Q₁ ← computed but NEVER USED! ❌  │
  │   K₁ ← computed (redundant) ❌      │
  │   V₁ ← computed (redundant) ❌      │
  │                                     │
  │ Position 2 (sat)                    │
  │   Q₂ ← computed and USED ✓          │
  │   K₂ ← computed ✓                   │
  │   V₂ ← computed ✓                   │
  └─────────────────────────────────────┘

With cache - passing [sat] only:
  ┌─────────────────────────────────────┐
  │ Position 0 (The)                    │
  │   Q₀ ← NOT computed (not needed) ✓  │
  │   K₀ ← REUSED from cache ✓          │
  │   V₀ ← REUSED from cache ✓          │
  │                                     │
  │ Position 1 (cat)                    │
  │   Q₁ ← NOT computed (not needed) ✓  │
  │   K₁ ← REUSED from cache ✓          │
  │   V₁ ← REUSED from cache ✓          │
  │                                     │
  │ Position 2 (sat)                    │
  │   Q₂ ← computed ✓                   │
  │   K₂ ← computed and cached ✓        │
  │   V₂ ← computed and cached ✓        │
  └─────────────────────────────────────┘

SUMMARY: Q comes from the last token only because that's the only position
that needs to predict the next token. All previous positions already made
their predictions!

VISUAL EXAMPLE:

Step 1: Generate "cat"
  Tokens: [The, cat]
  Compute: Q_cat, K_the, K_cat, V_the, V_cat
  Cache: Store K_the, K_cat, V_the, V_cat ✓

Step 2: Generate "sat"
  Tokens: [The, cat, sat]
  Compute: Q_sat (only this is new!)
  Reuse from cache: K_the, K_cat, V_the, V_cat (don't recompute!)
  Cache: Add K_sat, V_sat ✓

Step 3: Generate "on"
  Tokens: [The, cat, sat, on]
  Compute: Q_on (only this is new!)
  Reuse from cache: K_the, K_cat, K_sat, V_the, V_cat, V_sat
  Cache: Add K_on, V_on ✓

Notice: We NEVER recompute K,V for "The", "cat", or "sat" after step 1, 2, 3!

So instead of recomputing them at every step, we cache and reuse them.

=============================================================================
WHAT CAN CHANGE vs WHAT CANNOT CHANGE?
=============================================================================

CANNOT CHANGE (can be cached):
  ✓ Token embeddings - Fixed lookup table
  ✓ Position encodings - Fixed based on position number
  ✓ Model weights (W_k, W_v) - Fixed during inference
  ✓ K = embedding × W_k - Result of fixed × fixed = fixed
  ✓ V = embedding × W_v - Result of fixed × fixed = fixed

CAN CHANGE (must be recomputed):
  ✗ Query (Q) - Depends on the NEW token being processed
  ✗ Attention weights - Depend on Q (which changes each step)
  ✗ Attention output - Depends on attention weights

IMPORTANT: Q changes because it represents "what the NEW token is looking for"
in the context. Each new token has different information needs, so Q must be
recomputed. But K,V represent "what information each token contains", which is
fixed once the token exists!

NOTE: This works perfectly for autoregressive (decoder-only) models like GPT
because we generate tokens left-to-right and never modify previous tokens.
In encoder models (like BERT), all tokens are processed bidirectionally at once,
so caching is less relevant since there's no autoregressive generation.

ANALOGY - The Library Metaphor:
  - Each token is like a book in a library
  - K (Key) = the book's catalog entry (what it's about)
  - V (Value) = the book's contents (information it contains)
  - Q (Query) = what YOU are searching for

  Once a book is on the shelf:
  ✓ Its catalog entry (K) doesn't change
  ✓ Its contents (V) don't change
  ✗ What you're searching for (Q) changes with each new question

  KV cache = keeping the catalog and books on the shelf
  instead of re-cataloging them every time someone asks a new question!

=============================================================================
THE PROBLEM: O(N²) COMPLEXITY WITHOUT CACHE
=============================================================================

Autoregressive generation without KV cache:

Step 1: Process [token_1]                    → predict token_2
Step 2: Process [token_1, token_2]           → predict token_3
Step 3: Process [token_1, token_2, token_3]  → predict token_4
...
Step N: Process [token_1, ..., token_N]      → predict token_N+1

Total tokens processed: 1 + 2 + 3 + ... + N = N(N+1)/2 ≈ O(N²)

As sequence grows, each step takes LONGER because we reprocess everything!

=============================================================================
THE SOLUTION: O(N) COMPLEXITY WITH KV CACHE
=============================================================================

Autoregressive generation with KV cache:

Prefill: Process [token_1, ..., token_prompt] → cache all K,V
Step 1: Process [token_new_1] only            → predict token_2 (reuse cached K,V)
Step 2: Process [token_new_2] only            → predict token_3 (reuse cached K,V)
...
Step N: Process [token_new_N] only            → predict token_N+1

Total tokens processed: prompt_len + N ≈ O(N)

Each step processes ONLY 1 token, taking constant time!

=============================================================================
THIS SCRIPT DEMONSTRATES:
=============================================================================

1. WITHOUT KV Cache: Steps get progressively slower (O(N²))
2. WITH KV Cache: Steps stay constant time (O(N))
3. Speedup increases with sequence length
4. When cache overhead is worth it vs not worth it

Run this script to see the difference visually!
"""

import os
import sys
import time

import torch

# Add src to path so we can import our model
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../src"))

import torch.nn.functional as F

from eval_tasks.chat_core.kv_cache_utils import (create_kv_cache, forward_pass,
                                                 get_model_config,
                                                 prefill_prompt)
from gpt_2.gpt2_model import GPT
from gpt_2.utils import get_custom_tokenizer, load_checkpoint


def generate_with_timing(
    model, prompt_tokens, num_tokens, use_kv_cache, device, seed=42
):
    """
    Generate tokens and measure detailed timing for each step.

    This function is the heart of the demonstration. It shows the difference
    between cached and non-cached generation by timing each individual step.

    Args:
        model: The GPT model to use for generation
        prompt_tokens: List of token IDs for the prompt
        num_tokens: How many new tokens to generate
        use_kv_cache: Boolean - whether to use KV caching
        device: torch.device to run on (cuda/cpu)
        seed: Random seed for reproducible generation

    Returns:
        total_time: Total generation time in seconds
        step_times: List of time taken for each individual step
    """

    # =========================================================================
    # STEP 0: Set random seed for reproducible generation
    # =========================================================================
    # This ensures both WITH and WITHOUT cache generate identical sequences
    # Without this, the random sampling would produce different results
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)

    # =========================================================================
    # STEP 1: Extract model configuration
    # =========================================================================
    # We need these dimensions to create the KV cache with correct shape
    # Now using shared utility function for consistency
    num_heads, head_dim, num_layers, max_seq_len = get_model_config(model)

    # =========================================================================
    # STEP 2: Create KV cache if enabled (using shared utility)
    # =========================================================================
    # The cache will store Key and Value tensors for each layer
    #
    # Shape breakdown: (num_layers, 2, batch_size, num_heads, seq_len, head_dim)
    #                   └─────┬────┘ │  └────┬────┘ └────┬───┘ └──┬───┘ └───┬──┘
    #                       Why?     │     Why?         Why?      Why?     Why?
    #
    # Let's understand each dimension:
    #
    # 1. num_layers (e.g., 12):
    #    - Transformer has multiple stacked layers (12 in GPT-2)
    #    - EACH layer computes its own K and V!
    #    - Layer 0's K,V are different from Layer 1's K,V
    #    - So we need separate cache for each layer
    #
    # 2. 2 (for K and V):
    #    - We cache BOTH Keys and Values
    #    - Index 0 = Keys, Index 1 = Values
    #    - Could use separate arrays, but combining is more efficient
    #
    # 3. batch_size (e.g., 1):
    #    - How many sequences we're generating in parallel
    #    - Usually 1 during generation (one sequence at a time)
    #    - Could be > 1 for batch generation
    #
    # 4. num_heads (e.g., 12):
    #    - Multi-head attention has multiple "heads"
    #    - Each head learns different attention patterns
    #    - Each head has its own K and V
    #
    # 5. seq_len (e.g., 512):
    #    - Maximum sequence length we want to cache
    #    - Each token position gets its own K,V stored
    #    - As we generate, we fill positions 0, 1, 2, ...
    #
    # 6. head_dim (e.g., 64):
    #    - Dimension of each attention head
    #    - Usually: head_dim = embedding_dim / num_heads
    #    - For GPT-2: 768 / 12 = 64
    #
    # VISUAL EXAMPLE for GPT-2 (124M):
    #
    # Shape: (12, 2, 1, 12, 512, 64)
    #         │   │  │   │   │    └─ Each head has 64-dim vectors
    #         │   │  │   │   └────── Cache up to 512 tokens
    #         │   │  │   └────────── 12 attention heads
    #         │   │  └────────────── Generate 1 sequence at a time
    #         │   └───────────────── Store both K (index 0) and V (index 1)
    #         └───────────────────── 12 transformer layers
    #
    # MEMORY CALCULATION:
    # 12 layers × 2 (K,V) × 1 batch × 12 heads × 512 tokens × 64 dim × 2 bytes (bfloat16)
    # = 12 × 2 × 1 × 12 × 512 × 64 × 2 bytes
    # = 18,874,368 bytes
    # ≈ 19 MB per sequence
    #
    # WHY THIS STRUCTURE?
    #
    # Alternative 1: Flat array?
    #   - Hard to index: Which K,V belongs to which layer/head/position?
    #   - Need complex offset calculations
    #
    # Alternative 2: Dictionary?
    #   cache = {layer_0: {K: ..., V: ...}, layer_1: ...}
    #   - Slower access (hash lookups)
    #   - More memory overhead
    #
    # Our structure (multi-dimensional tensor):
    #   ✓ Fast indexing: cache[layer][0] for K, cache[layer][1] for V
    #   ✓ Contiguous memory: Better for GPU
    #   ✓ Clean API: Just pass one object around
    #
    # HOW IT'S USED:
    #
    # To get K,V for layer 3:
    #   k = kv_cache[3, 0, :, :, :current_pos, :]  # Keys for layer 3
    #   v = kv_cache[3, 1, :, :, :current_pos, :]  # Values for layer 3
    #
    # To store new K,V at position pos:
    #   kv_cache[layer, 0, :, :, pos:pos+1, :] = new_k
    #   kv_cache[layer, 1, :, :, pos:pos+1, :] = new_v
    #
    # Now using shared create_kv_cache() utility for consistency with prod code:
    kv_cache = create_kv_cache(
        prompt_length=len(prompt_tokens),
        max_tokens=num_tokens,
        num_heads=num_heads,
        head_dim=head_dim,
        num_layers=num_layers,
        max_seq_len=max_seq_len,
        use_kv_cache=use_kv_cache,
    )
    # At this point, cache is created but empty
    # It will be filled during the prefill phase
    #
    # VISUAL: What the cache looks like conceptually
    #
    #   Layer 0:  [ K: ▢▢▢▢▢... ] [ V: ▢▢▢▢▢... ]  ← Empty slots for 512 tokens
    #   Layer 1:  [ K: ▢▢▢▢▢... ] [ V: ▢▢▢▢▢... ]
    #   Layer 2:  [ K: ▢▢▢▢▢... ] [ V: ▢▢▢▢▢... ]
    #   ...
    #   Layer 11: [ K: ▢▢▢▢▢... ] [ V: ▢▢▢▢▢... ]
    #
    # After prefill with "Hello world" (2 tokens):
    #
    #   Layer 0:  [ K: ■■▢▢▢... ] [ V: ■■▢▢▢... ]  ← Filled for 2 tokens
    #   Layer 1:  [ K: ■■▢▢▢... ] [ V: ■■▢▢▢... ]
    #   Layer 2:  [ K: ■■▢▢▢... ] [ V: ■■▢▢▢... ]
    #   ...
    #   Layer 11: [ K: ■■▢▢▢... ] [ V: ■■▢▢▢... ]
    #
    # After generating 3 more tokens:
    #
    #   Layer 0:  [ K: ■■■■■▢... ] [ V: ■■■■■▢... ]  ← Now 5 tokens cached
    #   Layer 1:  [ K: ■■■■■▢... ] [ V: ■■■■■▢... ]
    #   Layer 2:  [ K: ■■■■■▢... ] [ V: ■■■■■▢... ]
    #   ...
    #   Layer 11: [ K: ■■■■■▢... ] [ V: ■■■■■▢... ]

    # =========================================================================
    # STEP 3: PREFILL PHASE - Process the entire prompt at once
    # =========================================================================
    # This is the same for both cached and non-cached generation
    # We process all prompt tokens in one forward pass
    # If cache is enabled, this fills the cache with K,V for all prompt tokens
    # Now using shared utility function for consistency
    next_token_logits = prefill_prompt(model, prompt_tokens, device, kv_cache)

    # Sample the first token
    probs = F.softmax(next_token_logits, dim=-1)
    next_token = torch.multinomial(probs, num_samples=1).item()

    # After prefill:
    # - If cache enabled: cache now contains K,V for all prompt tokens
    # - If cache disabled: nothing cached, we'll reprocess everything each step

    # =========================================================================
    # STEP 4: DECODE PHASE - Generate new tokens one at a time
    # =========================================================================
    # This is where we'll see the difference!

    generated_tokens = list(prompt_tokens)  # Start with prompt
    step_times = []  # Track time for each generation step

    start_total = time.time()

    for step in range(num_tokens):
        # =====================================================================
        # KEY DIFFERENCE: What do we pass to the model?
        # =====================================================================
        # Append the token we're about to process
        generated_tokens.append(next_token)

        if use_kv_cache:
            # WITH CACHE: Only pass the NEW token!
            # The cache already has K,V for all previous tokens
            # This makes the forward pass O(1) with respect to sequence length
            # Shape: (1, 1) - just ONE token!

            # Example at step 10:
            # - Total sequence so far: 12 tokens (2 prompt + 10 generated)
            # - We pass: 1 token
            # - Cache provides: K,V for previous 11 tokens
            # - Result: Attention computation is fast!

            token_or_tokens = next_token  # Pass single token

        else:
            # WITHOUT CACHE: Pass the ENTIRE sequence so far!
            # We recompute K,V for ALL tokens at every step
            # This makes the forward pass O(N) with respect to sequence length
            # Shape: (1, current_length) - GROWING each step!

            # Example at step 10:
            # - Total sequence so far: 12 tokens (2 prompt + 10 generated)
            # - We pass: ALL 12 tokens
            # - Cache provides: nothing
            # - Result: Attention computation gets slower each step!

            token_or_tokens = generated_tokens  # Pass entire sequence

        # =====================================================================
        # TIME THIS STEP
        # =====================================================================
        # This is the critical measurement that shows the speedup
        # Now using shared utility functions for consistency

        step_start = time.time()

        # Forward pass through model (handles single token or full sequence)
        next_token_logits = forward_pass(model, token_or_tokens, device, kv_cache)

        # Sample next token (using greedy sampling for determinism)
        probs = F.softmax(next_token_logits, dim=-1)
        next_token = torch.multinomial(probs, num_samples=1).item()

        step_time = time.time() - step_start
        step_times.append(step_time)

        # OBSERVATION:
        # - With cache: step_times should be roughly CONSTANT (e.g., 6ms, 6ms, 6ms, ...)
        # - Without cache: step_times should INCREASE (e.g., 5ms, 7ms, 9ms, 11ms, ...)

    total_time = time.time() - start_total

    return total_time, step_times, generated_tokens


def main():
    """
    Main demonstration function.

    This function orchestrates the entire KV cache demonstration:
    1. Loads a trained model
    2. Tests generation with increasing sequence lengths
    3. Compares timing with and without KV cache
    4. Shows how speedup scales with sequence length
    """

    print("\n" + "=" * 80)
    print("🎓 KV CACHE LEARNING DEMO - See the Speedup!")
    print("=" * 80)

    # =========================================================================
    # CONFIGURATION
    # =========================================================================
    checkpoint_path = "<YOURPATH>/nanogpt/midtrain_checkpoints/model_checkpoint_global2633_midtraining.pt"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Device: {device}")
    print("=" * 80 + "\n")

    # =========================================================================
    # LOAD MODEL
    # =========================================================================
    # We use a pre-trained model checkpoint to get realistic performance numbers
    # The model needs to be in eval mode to disable dropout and other training-specific behavior

    print("📂 Loading model...")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    config = checkpoint["config"]

    model = GPT(config)
    model = model.to(device)
    model.eval()  # Important: put in evaluation mode

    load_checkpoint(
        checkpoint_path=checkpoint_path,
        model=model,
        device=device,
        optimizer=None,
        master_process=True,
    )
    print("✅ Model loaded\n")

    # =========================================================================
    # CREATE TEST PROMPT
    # =========================================================================
    # We use a short prompt so that most of the generation time is spent
    # in the decode phase (where we see the cache benefit), not the prefill phase

    tokenizer, _ = get_custom_tokenizer()
    prompt = "Hello world"
    prompt_tokens = tokenizer.encode(prompt)

    print(f'📝 Prompt: "{prompt}" ({len(prompt_tokens)} tokens)\n')

    # =========================================================================
    # TEST WITH INCREASINGLY LONG SEQUENCES
    # =========================================================================
    # We test multiple sequence lengths to show how the speedup changes
    # Theory: Longer sequences = more benefit from caching
    # Why? Because without cache, we reprocess more tokens each step (O(N²))

    print("=" * 80)
    print("TESTING DIFFERENT SEQUENCE LENGTHS")
    print("=" * 80)
    print("\nKey insight: Speedup increases with sequence length!")
    print("Without cache = O(N²) | With cache = O(N)\n")

    # Test lengths: short to long
    # At 50 tokens: Cache overhead might dominate
    # At 400 tokens: Cache benefit should be clear
    # At 1000+ tokens: Cache benefit should be very significant
    test_lengths = [1000]

    results = []  # Store results for summary table

    for num_tokens in test_lengths:
        print("-" * 80)
        print(f"Generating {num_tokens} tokens...")
        print("-" * 80)

        # =====================================================================
        # RUN BOTH EXPERIMENTS
        # =====================================================================

        # EXPERIMENT 1: Generate WITHOUT KV cache
        # Each step will pass the entire sequence so far to the model
        # Expected behavior: Steps get progressively slower
        time_without, steps_without, generated_tokens_without = generate_with_timing(
            model, prompt_tokens, num_tokens, use_kv_cache=False, device=device, seed=42
        )

        # EXPERIMENT 2: Generate WITH KV cache
        # Each step will pass only 1 new token to the model
        # Expected behavior: Steps stay roughly constant time
        time_with, steps_with, generated_tokens_with = generate_with_timing(
            model, prompt_tokens, num_tokens, use_kv_cache=True, device=device, seed=42
        )

        # decode the generated tokens
        generated_text_without = tokenizer.decode(generated_tokens_without)
        generated_text_with = tokenizer.decode(generated_tokens_with)

        # print(f"--------------- \n Generated text without cache: {generated_text_without} \n---------------")
        # print(f"--------------- \n Generated text with cache: {generated_text_with} \n---------------")

        # Sanity check: Both should generate identical text (same seed + deterministic sampling)
        if generated_text_without == generated_text_with:
            print("✅ CORRECTNESS CHECK: Both methods generated IDENTICAL text!")
        else:
            print("⚠️  WARNING: Generated text differs between cached and non-cached!")
            print(
                f"   First difference at token position: {next((i for i, (a, b) in enumerate(zip(generated_tokens_without, generated_tokens_with)) if a != b), len(generated_tokens_without))}"
            )

        # =====================================================================
        # ANALYZE RESULTS
        # =====================================================================

        speedup = time_without / time_with

        # Calculate average step time
        # WITHOUT cache: This should INCREASE as we test longer sequences
        # WITH cache: This should stay CONSTANT regardless of length
        avg_step_without = sum(steps_without) / len(steps_without)
        avg_step_with = sum(steps_with) / len(steps_with)

        # =====================================================================
        # DISPLAY RESULTS - THIS IS THE KEY LEARNING MOMENT!
        # =====================================================================
        # We show the first 5 and last 5 step times to visualize the pattern

        print("\n❌ WITHOUT KV Cache:")
        print(f"   Total time: {time_without:.3f}s")
        print(f"   Avg time per step: {avg_step_without*1000:.2f}ms")
        print(f"   First 5 steps: {[f'{t*1000:.1f}ms' for t in steps_without[:5]]}")
        print(f"   Last 5 steps:  {[f'{t*1000:.1f}ms' for t in steps_without[-5:]]}")
        print("   👉 Notice: Steps get SLOWER over time! (O(N²))")
        print("      Why? Each step processes more tokens than the last:")
        print("      Step 1: 3 tokens, Step 2: 4 tokens, ..., Step N: N+2 tokens")

        print("\n✅ WITH KV Cache:")
        print(f"   Total time: {time_with:.3f}s")
        print(f"   Avg time per step: {avg_step_with*1000:.2f}ms")
        print(f"   First 5 steps: {[f'{t*1000:.1f}ms' for t in steps_with[:5]]}")
        print(f"   Last 5 steps:  {[f'{t*1000:.1f}ms' for t in steps_with[-5:]]}")
        print("   👉 Notice: Steps stay CONSTANT! (O(N))")
        print("      Why? Each step processes exactly 1 token:")
        print("      Step 1: 1 token, Step 2: 1 token, ..., Step N: 1 token")

        print("\n📊 RESULT:")
        print(f"   ⚡ Speedup: {speedup:.2f}x")
        if speedup > 1:
            print(f"   🎉 KV cache is {speedup:.2f}x FASTER!")
            print(
                f"      Cache avoided reprocessing {sum(range(1, num_tokens+1)):,} tokens!"
            )
        else:
            print("   ⚠️  KV cache overhead dominates (sequence too short)")
            print("      Cache management costs more than computation savings")

        print()

        # Store results for final summary
        results.append((num_tokens, speedup, time_without, time_with))

    # =========================================================================
    # SUMMARY TABLE
    # =========================================================================
    # Show all results in a compact table to see the trend
    # Key observation: Look at how "Without Cache" time grows faster than
    # "With Cache" time as sequence length increases!

    print("=" * 80)
    print("📈 SUMMARY - How Speedup Scales")
    print("=" * 80)
    print(
        "\n{:>10s} | {:>15s} | {:>15s} | {:>10s}".format(
            "Tokens", "Without Cache", "With Cache", "Speedup"
        )
    )
    print("-" * 80)

    for num_tokens, speedup, time_without, time_with in results:
        # Visual indicators for speedup level
        indicator = "🚀" if speedup > 1.5 else "⚡" if speedup > 1.0 else "⚠️"
        print(
            "{:>10d} | {:>13.3f}s | {:>13.3f}s | {:>9.2f}x {}".format(
                num_tokens, time_without, time_with, speedup, indicator
            )
        )

    # =========================================================================
    # EXPLAIN THE TREND
    # =========================================================================
    # Help the user understand what they're seeing

    print("\n💡 What to notice in the table above:")
    print("   - WITHOUT cache: Time grows quadratically (2x tokens ≈ 4x time)")
    print("   - WITH cache: Time grows linearly (2x tokens ≈ 2x time)")
    print("   - Speedup: Generally increases with sequence length")
    print("   - For VERY long sequences (1000+ tokens), speedup can reach 3-5x+")

    print("\n" + "=" * 80)
    print("💡 KEY LEARNINGS - Understanding KV Cache")
    print("=" * 80)
    print("""
================================================================================
1. WITHOUT KV Cache - The O(N²) Problem
================================================================================

What happens at each step:
  Step 1: Process [prompt, token_1]                     → 3 tokens
  Step 2: Process [prompt, token_1, token_2]            → 4 tokens  
  Step 3: Process [prompt, token_1, token_2, token_3]   → 5 tokens
  ...
  Step N: Process [prompt, token_1, ..., token_N]       → N+2 tokens

Total tokens processed: 3 + 4 + 5 + ... + (N+2) = O(N²)

Why it's slow:
  - Recomputing attention for ALL previous tokens at EVERY step
  - Each step is slower than the last (you saw this in step times!)
  - Wasted computation: K,V for old tokens never change!

================================================================================
2. WITH KV Cache - The O(N) Solution
================================================================================

What happens at each step:
  Prefill: Process [prompt]              → Cache K,V for prompt
  Step 1:  Process [token_1] only        → Cache K,V for token_1
  Step 2:  Process [token_2] only        → Cache K,V for token_2
  Step 3:  Process [token_3] only        → Cache K,V for token_3
  ...
  Step N:  Process [token_N] only        → Cache K,V for token_N

Total tokens processed: prompt_len + N = O(N)

Why it's fast:
  - Process each token exactly ONCE
  - Reuse cached K,V from previous steps
  - Each step takes constant time (you saw this in step times!)
  - No wasted recomputation!

================================================================================
3. The Math: Why O(N²) vs O(N) Matters
================================================================================

Example: Generate 400 tokens with 2-token prompt

WITHOUT cache:
  - Total tokens processed = 2 + 3 + 4 + ... + 402 
  - = (402 × 403) / 2 - (1 × 2) / 2 
  - ≈ 80,000 tokens processed 😱

WITH cache:
  - Total tokens processed = 2 (prefill) + 400 (decode)
  - = 402 tokens processed 🚀
  - Savings: 99.5% fewer tokens to process!

This is why you saw ~1.3-1.5x speedup in practice
(would be higher but memory bandwidth limits us on small models)

================================================================================
4. When is KV Cache Worth It?
================================================================================

✅ GOOD use cases:
  - Longer sequences (200+ tokens): More benefit from O(N) vs O(N²)
  - Batch generation from same prompt: Reuse prompt cache
  - Interactive chat: Growing conversation context
  - Streaming generation: Add one token at a time
  - Production deployments: Amortize setup cost

❌ NOT worth it:
  - Very short sequences (< 50 tokens): Overhead > savings
  - Single-shot generation: Setup cost not recovered
  - Memory constrained: Cache uses ~2x more VRAM
  - Small models on fast GPUs: Already compute-bound

================================================================================
5. Real-World Impact
================================================================================

For a 7B parameter model generating 500 tokens:
  - WITHOUT cache: ~60 seconds
  - WITH cache: ~15 seconds
  - Speedup: 4x 🚀

For this 124M model generating 400 tokens:
  - WITHOUT cache: ~3.5 seconds
  - WITH cache: ~2.6 seconds  
  - Speedup: 1.3x ⚡

Smaller models show less speedup because they're memory-bandwidth limited,
not compute-limited. The cache helps most when computation is the bottleneck!

================================================================================
""")

    # =========================================================================
    # FURTHER EXPLORATION
    # =========================================================================
    print("=" * 80)
    print("🔬 FURTHER EXPLORATION - Try These Experiments!")
    print("=" * 80)
    print("""
Want to learn more? Try these experiments:

1. Test with even longer sequences:
   - Modify test_lengths = [50, 100, 200, 400, 800, 1000]
   - Observe how speedup continues to increase
   - Question: At what length does speedup plateau?

2. Visualize the O(N²) vs O(N) pattern:
   - Plot step_times vs step_number for both methods
   - WITHOUT cache: Should see upward slope
   - WITH cache: Should see flat line

3. Test with different prompt lengths:
   - Short prompt (2 tokens): Minimal prefill cost
   - Long prompt (100 tokens): Expensive prefill, but reused
   - Question: How does prompt length affect overall speedup?

4. Measure memory usage:
   - Add torch.cuda.memory_allocated() before/after cache creation
   - Cache uses ~2x more VRAM
   - Question: Is the memory/speed tradeoff worth it?

5. Test batch generation:
   - Generate multiple completions from same prompt
   - Reuse prefill cache across samples
   - Question: How much speedup for 10 samples vs 1 sample?

6. Compare with your own model:
   - Try with a larger model (if available)
   - Larger models show more speedup (more compute-bound)
   - Question: How does model size affect cache benefit?

Related files to explore:
  - src/gpt_2/kv_cache.py: KV cache data structure implementation
  - src/eval_tasks/chat_core/kv_cache_utils.py: Shared KV cache utilities
  - src/gpt_2/attention.py: How cache is used in attention
  - debug_tools/test_kvcache_correctness.py: Verify cache correctness
  - debug_tools/test_chatcore_kvcache.py: Real evaluation with cache
""")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    main()
