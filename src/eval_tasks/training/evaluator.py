"""
Training Evaluator for validation loss computation and model sampling during training.

This module provides:
- TrainingEvaluator: Computes validation loss/BPB and samples from the model
- generate(): Text generation function with KV caching support
"""

import math
import time

import torch
import torch.distributed as dist
import torch.nn.functional as F

import wandb
from gpt_2.kv_cache import KVCache
from gpt_2.utils import accumulate_bpb, get_custom_tokenizer


def generate(
    num_sequences,
    max_length,
    model,
    context,
    device,
    random_number_generator,
    use_kv_cache=True,
):
    """
    Generate text sequences using the trained GPT model with optional KV caching.

    This function performs autoregressive text generation using top-k sampling
    to produce diverse and coherent text continuations.

    KV Caching: When enabled (default), uses a two-phase approach:
    1. PREFILL: Process all context tokens at once, cache their K/V
    2. DECODE: Generate one token at a time, reusing cached K/V
    This reduces computation from O(N²) to O(N), making generation 3-10x faster.

    Args:
        num_sequences (int): Number of sequences to generate
        max_length (int): Maximum length of each generated sequence
        model (GPT): The trained GPT model
        context (str): Initial text context to start generation from
        device (str): Device to run generation on ('cuda', 'mps', or 'cpu')
        random_number_generator: PyTorch random generator for sampling
        use_kv_cache (bool): Whether to use KV caching (default: True)

    Returns:
        list: List of decoded text sequences
    """
    # Initialize the custom tokenizer with special tokens for chat format
    enc, _ = get_custom_tokenizer()

    # Encode the context string to token indices (allow special tokens in context)
    tokens = enc.encode(context, allowed_special="all")
    tokens = torch.tensor(tokens, dtype=torch.long)

    # Create multiple copies for batch generation
    tokens = tokens.unsqueeze(0).repeat(num_sequences, 1)
    x = tokens.to(device)

    print(f"\n{'='*60}")
    print("Starting generation:")
    print(f"  - Number of sequences: {num_sequences}")
    print(f"  - Context length: {x.size(1)} tokens")
    print(f"  - Target max length: {max_length} tokens")
    print(f"  - KV cache enabled: {use_kv_cache}")
    print(f"{'='*60}\n")

    # Set random seed for reproducible generation
    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    # =========================================================================
    # SETUP: Create KV cache if enabled
    # =========================================================================
    kv_cache = None
    if use_kv_cache:
        # Extract model dimensions for KV cache
        config = model.config
        num_heads = config.n_kv_head if hasattr(config, "n_kv_head") else config.n_head
        head_dim = config.n_embed // config.n_head
        num_layers = config.n_layer

        # =====================================================================
        # OPTIMIZATION: Use prefill pattern for batch generation
        # =====================================================================
        # For batch generation from the same prompt, we can:
        # 1. Process prompt ONCE with batch_size=1 (efficient single forward pass)
        # 2. Copy cached K/V to batch_size=num_sequences (cheap memory copy)
        # 3. Generate multiple sequences in parallel
        # This avoids recomputing the same prompt num_sequences times!

        if num_sequences > 1:
            # Step 1: Create single-batch cache and process prompt once
            single_cache = KVCache(
                batch_size=1,
                num_heads=num_heads,
                seq_len=max_length,
                head_dim=head_dim,
                num_layers=num_layers,
            )

            # Process prompt once with batch_size=1
            prompt_tensor = x[0:1, :]  # Take first sequence (they're all identical)
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits_single, _ = model(prompt_tensor, kv_cache=single_cache)

            # Step 2: Create batch cache and prefill from single cache
            kv_cache = KVCache(
                batch_size=num_sequences,
                num_heads=num_heads,
                seq_len=max_length,
                head_dim=head_dim,
                num_layers=num_layers,
            )
            kv_cache.prefill(single_cache)  # Copy cached K/V to all batch positions

            # Replicate logits for all sequences
            logits = logits_single.repeat(num_sequences, 1, 1)  # (B, T, vocab_size)
            print(f"✓ Prefill complete: Processed {x.size(1)} context tokens")
        else:
            # Single sequence: no need for prefill optimization
            kv_cache = KVCache(
                batch_size=num_sequences,
                num_heads=num_heads,
                seq_len=max_length,
                head_dim=head_dim,
                num_layers=num_layers,
            )

            # Process prompt tokens
            with torch.no_grad():
                with torch.autocast(device_type=device, dtype=torch.bfloat16):
                    logits, _ = model(x, kv_cache=kv_cache)  # Shape: (B, T, vocab_size)
            print(f"✓ Prefill complete: Processed {x.size(1)} context tokens")
    else:
        # =====================================================================
        # NO KV CACHE: Process entire sequence each time (O(N²))
        # =====================================================================
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                logits, _ = model(x, kv_cache=None)  # Shape: (B, T, vocab_size)

    # Only use the last token's predictions for next token
    logits = logits[:, -1, :]  # Shape: (B, vocab_size)

    # =========================================================================
    # PHASE 2: DECODE - Generate tokens one at a time
    # =========================================================================
    # Generate tokens autoregressively until max_length is reached
    print("Starting token generation (decode phase)...\n")
    tokens_to_generate = max_length - x.size(1)
    while x.size(1) < max_length:
        # Convert logits to probabilities
        probs = F.softmax(logits, dim=-1)  # Shape: (B, vocab_size)

        # Apply top-k sampling (k=50) for diverse generation
        # This helps avoid repetitive text by sampling from top-k most likely tokens
        topk_probs, topk_indices = torch.topk(
            probs, 50, dim=-1
        )  # topk_probs: (B, 50), topk_indices: (B, 50)

        # Sample from the top-k distribution
        ix = torch.multinomial(
            topk_probs, num_samples=1, generator=random_number_generator
        )  # ix: (B, 1)

        # Get the actual token indices from the top-k indices
        xcol = torch.gather(topk_indices, -1, ix)  # Shape: (B, 1)

        # Append the new token to the sequence
        x = torch.cat((x, xcol), dim=1)

        # Progress monitoring
        current_len = x.size(1)
        tokens_generated = current_len - (max_length - tokens_to_generate)
        if tokens_generated % 10 == 0 or current_len == max_length:
            progress_pct = (tokens_generated / tokens_to_generate) * 100
            print(
                f"  Progress: {tokens_generated}/{tokens_to_generate} tokens generated ({progress_pct:.1f}%)"
            )

        # Get next token's logits
        with torch.no_grad():
            with torch.autocast(device_type=device, dtype=torch.bfloat16):
                if use_kv_cache:
                    # WITH CACHE: Only pass the new token (O(1) per step)
                    logits, _ = model(xcol, kv_cache=kv_cache)
                else:
                    # WITHOUT CACHE: Pass entire sequence (O(N) per step)
                    logits, _ = model(x, kv_cache=None)

        # Extract logits for next prediction
        logits = logits[:, -1, :]  # Shape: (B, vocab_size)

    # Decode and return all generated sequences
    print(f"\n{'='*60}")
    print("Generation complete!")
    print(f"  - Generated {num_sequences} sequence(s)")
    print(f"  - Final length: {x.size(1)} tokens")
    print(f"{'='*60}\n")

    all_decoded = []
    for i in range(num_sequences):
        tokens = x[i, :max_length].tolist()
        decoded = enc.decode(tokens)
        all_decoded.append(decoded)
    return all_decoded


class TrainingEvaluator:
    """
    Evaluator for training-time validation and sampling.

    This evaluator:
    1. Computes validation loss and bits per byte (BPB) on validation data
    2. Generates sample text during training for qualitative evaluation
    3. Logs results to wandb
    """

    def __init__(
        self,
        model,
        eval_dataloader,
        device,
        master_process,
        ddp,
        ddp_rank=0,
        ddp_world_size=1,
        generation_log_file=None,
        token_bytes_path=None,
        val_loss_steps=39,
        sample_seed=42,
        use_kv_cache=True,
    ):
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.master_process = master_process
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.generation_log_file = generation_log_file
        self.val_loss_steps = val_loss_steps
        self.sample_seed = sample_seed
        self.use_kv_cache = use_kv_cache

        # Load pre-computed token_bytes tensor for BPB calculation
        if token_bytes_path is not None:
            self._token_bytes = torch.load(token_bytes_path, weights_only=True).to(
                self.device
            )
        else:
            raise ValueError("token_bytes_path is required for BPB calculation")

    def estimate_validation_loss(self, step, global_step=None):
        """
        Estimate average loss on validation set and compute bits per byte (BPB).
        BPB is tokenization-independent: normalized by actual byte length of tokens.

        Args:
            step: Current step within epoch
            global_step: Global step across all epochs (for wandb logging)

        Returns:
            float: Validation loss
        """
        start_time = time.time()

        self.model.eval()
        self.eval_dataloader.reset()
        val_loss_steps = self.val_loss_steps

        token_bytes = self._token_bytes
        # Accumulators for loss and BPB
        val_loss_accumulator = torch.tensor(0.0, device=self.device)
        total_nats = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        total_bytes = torch.tensor(0, dtype=torch.int64, device=self.device)

        with torch.no_grad():
            for k in range(val_loss_steps):
                X, Y = self.eval_dataloader.next_batch()
                X = X.to(self.device)
                Y = Y.to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    _, per_token_loss = self.model(
                        X, Y, loss_reduction="none"
                    )  # (B*T,)

                # Compute mean loss from per-token losses
                val_loss_accumulator += per_token_loss.mean() / val_loss_steps

                # BPB calculation
                nats, bytes = accumulate_bpb(per_token_loss, Y, token_bytes)
                total_nats += nats
                total_bytes += bytes

        self.model.train()

        # Reduce across ranks
        if self.ddp_world_size > 1:
            dist.all_reduce(val_loss_accumulator, op=dist.ReduceOp.AVG)
            dist.all_reduce(total_nats, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_bytes, op=dist.ReduceOp.SUM)

        elapsed_time = time.time() - start_time

        # Calculate BPB
        val_loss = val_loss_accumulator.item()
        total_nats_val = total_nats.item()
        total_bytes_val = total_bytes.item()
        if total_bytes_val == 0:
            val_bpb = float("inf")
        else:
            val_bpb = total_nats_val / (math.log(2) * total_bytes_val)

        if self.master_process:
            print(f"\n{'='*80}")
            print(
                f"📊 VALIDATION | Step {step:>5} | Val Loss: {val_loss:.4f} | BPB: {val_bpb:.4f} | Time: {elapsed_time:.2f}s"
            )
            print(f"{'='*80}\n")
            wandb.log(
                {
                    "val_loss": val_loss,
                    "val_bpb": val_bpb,
                    "step": global_step if global_step is not None else step,
                }
            )

        return val_loss

    def sample_from_model(
        self,
        num_sequences=4,
        max_length=255,
        context="Hello, I'm a language model,",
        step=None,
    ):
        if not self.master_process:
            return

        start_time = time.time()

        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(self.sample_seed + self.ddp_rank)

        cache_status = "with KV cache" if self.use_kv_cache else "without KV cache"
        print(
            f"Generating {num_sequences} sequences of length {max_length} {cache_status} | Context: {context}"
        )

        # Unwrap model from DDP if needed (generation only happens on master process)
        raw_model = self.model.module if self.ddp else self.model

        decoded = generate(
            num_sequences=num_sequences,
            max_length=max_length,
            model=raw_model,
            context=context,
            device=self.device,
            random_number_generator=sample_rng,
            use_kv_cache=self.use_kv_cache,
        )

        elapsed_time = time.time() - start_time

        # Print to console (truncated)
        print(f"🎯 SAMPLE GENERATIONS (Time: {elapsed_time:.2f}s):")
        for i, decoded_seq in enumerate(decoded, 1):
            # Truncate if too long and add ellipsis
            display_text = (
                decoded_seq if len(decoded_seq) <= 200 else decoded_seq[:200] + "..."
            )
            print(f"  {i}. {display_text}")
        print()

        # Save full generations to log file
        if self.generation_log_file:
            from datetime import datetime

            with open(self.generation_log_file, "a") as f:
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                step_info = f"Step {step}" if step is not None else "Unknown Step"
                f.write(f"\n{'='*80}\n")
                f.write(f"[{timestamp}] {step_info}\n")
                f.write(f"Context: {context}\n")
                f.write(f"{'-'*80}\n")
                for i, decoded_seq in enumerate(decoded, 1):
                    f.write(f"\nGeneration {i}:\n")
                    f.write(f"{decoded_seq}\n")
                    if i < len(
                        decoded
                    ):  # Add separator between generations, but not after last one
                        f.write(f"{'-'*80}\n")
                f.write(f"{'='*80}\n")
