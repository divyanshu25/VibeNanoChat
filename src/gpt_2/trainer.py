# Add gpt_2 to python path
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import random
import time

import torch

import wandb
from gpt_2.muon import get_muon_momentum, get_muon_weight_decay
from gpt_2.sample_contexts import GENERAL_SAMPLE_CONTEXTS, SFT_SAMPLE_CONTEXTS
from gpt_2.training_utilities import (setup_dataloaders, setup_hyperparameters,
                                      setup_logging, setup_model, setup_wandb)
from gpt_2.utils import load_checkpoint, save_checkpoint


class Trainer:
    """
    GPT-2 Trainer class that handles model training, evaluation, and optimization.
    Implements modern training techniques like learning rate scheduling and gradient clipping.
    """

    def __init__(
        self,
        ddp,
        ddp_rank,
        ddp_local_rank,
        ddp_world_size,
        master_process,
        device,
        run_evals=False,
        run_core_evals=False,
        run_chatcore_evals=False,
        sft_training=False,
        checkpoint_path=None,
        checkpoint_dir=None,
        token_bytes_path=None,
        depth=None,
        aspect_ratio=None,
        head_dim=None,
        target_flops=None,
        param_data_ratio=None,
        eval_interval=None,
        core_eval_interval=None,
    ):
        """
        Initialize trainer with model configuration, data loading, and training parameters.

        Args:
            ddp: Whether to use distributed data parallel
            ddp_rank: Rank of current process
            ddp_local_rank: Local rank of current process
            ddp_world_size: Total number of processes
            master_process: Whether this is the master process
            device: Device to train on
            run_evals: Whether to run evaluations
            run_core_evals: Whether to run CORE benchmark evaluations
            run_chatcore_evals: Whether to run ChatCORE generative evaluations (GSM8K, etc.) after training
            sft_training: Whether to do SFT training (uses Multiplex dataloader with conversation data)
            checkpoint_path: Path to checkpoint to load (for SFT or resuming)
            checkpoint_dir: Directory to save checkpoints (pretraining or SFT specific)
            token_bytes_path: Path to pre-computed token_bytes.pt for BPB calculation
            depth: Model depth (auto-calculates n_layer/n_embed/n_head from depth × aspect_ratio)
            aspect_ratio: Aspect ratio for depth mode (model_dim = depth × aspect_ratio, default from config)
            head_dim: Target head dimension for depth mode (default from config)
            target_flops: Target total FLOPs for training (overrides config.target_flops)
            param_data_ratio: Token:Param ratio for training (overrides config.target_param_data_ratio)
        """
        # Store DDP (Distributed Data Parallel) configuration
        # DDP allows training across multiple GPUs/nodes by splitting batches
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.device = device
        self.master_process = (
            master_process  # Only master process does logging/checkpointing
        )

        # Store training mode flags
        self.run_evals = run_evals
        self.run_core_evals = run_core_evals
        self.run_chatcore_evals = run_chatcore_evals
        self.sft_training = sft_training

        # Store checkpoint paths
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = checkpoint_dir
        self.token_bytes_path = token_bytes_path

        # Store model architecture overrides (for sweeps/experiments)
        self.depth_override = depth
        self.aspect_ratio_override = aspect_ratio
        self.head_dim_override = head_dim
        self.target_flops_override = target_flops
        self.param_data_ratio_override = param_data_ratio
        self.eval_interval_override = eval_interval
        self.core_eval_interval_override = core_eval_interval

        # Training state counters - will be set to nonzero if resuming from checkpoint
        self.start_step = 0
        self.start_epoch = 0
        self.start_global_step = 0

        # Pick random sample contexts to evaluate model's generation quality during training
        # This gives qualitative insight into what the model is learning beyond loss numbers
        if self.sft_training:
            self.sample_contexts = random.sample(
                SFT_SAMPLE_CONTEXTS, min(4, len(SFT_SAMPLE_CONTEXTS))
            )
        else:
            self.sample_contexts = random.sample(
                GENERAL_SAMPLE_CONTEXTS, min(4, len(GENERAL_SAMPLE_CONTEXTS))
            )

        # Setup model (includes architecture, wrapping with DDP if needed)
        self.config, self.raw_model, self.model = setup_model(
            depth_override=self.depth_override,
            aspect_ratio_override=self.aspect_ratio_override,
            head_dim_override=self.head_dim_override,
            target_flops_override=self.target_flops_override,
            param_data_ratio_override=self.param_data_ratio_override,
            eval_interval_override=self.eval_interval_override,
            device=self.device,
            ddp=self.ddp,
            master_process=self.master_process,
        )

        # Setup logging (need model first to get depth for filename)
        self.generation_log_file = setup_logging(
            self.master_process,
            depth=self.config.depth if self.config._depth_mode else None,
            sft_training=self.sft_training,
        )

        # Calculate hyperparameters (batch size, learning rates, schedule, etc)
        hyperparams = setup_hyperparameters(
            config=self.config,
            raw_model=self.raw_model,
            sft_training=self.sft_training,
            ddp_world_size=self.ddp_world_size,
            device=self.device,
            eval_interval_override=self.eval_interval_override,
            core_eval_interval_override=self.core_eval_interval_override,
            master_process=self.master_process,
        )

        # Unpack hyperparameters into instance variables for easy access
        self.num_epochs = hyperparams["num_epochs"]
        self.grad_accumulation_steps = hyperparams["grad_accumulation_steps"]
        self.total_batch_size = hyperparams["total_batch_size"]
        self.max_steps = hyperparams["max_steps"]
        self.flops_per_token = hyperparams["flops_per_token"]
        self.peak_flops = hyperparams["peak_flops"]
        self.run_evals_after = hyperparams["run_evals_after"]
        self.run_core_evals_after = hyperparams["run_core_evals_after"]

        # Setup remaining components (dataloaders, optimizer, wandb)
        self._setup_dataloaders()
        self._setup_optimizer_and_checkpoint()
        self._setup_wandb()

    def _setup_dataloaders(self):
        """
        Setup training and evaluation dataloaders.

        Training dataloader yields infinite stream of batches (wraps around dataset).
        Evaluation dataloaders are used periodically to estimate val loss and run benchmarks.
        """
        results = setup_dataloaders(
            sft_training=self.sft_training,
            config=self.config,
            ddp_world_size=self.ddp_world_size,
            ddp_rank=self.ddp_rank,
            master_process=self.master_process,
            run_evals=self.run_evals,
            run_core_evals=self.run_core_evals,
            run_chatcore_evals=self.run_chatcore_evals,
            raw_model=self.raw_model,
            device=self.device,
            ddp=self.ddp,
            generation_log_file=self.generation_log_file,
            token_bytes_path=self.token_bytes_path,
        )
        (
            self.train_dataloader,
            self.evaluator,
            self.core_evaluator,
            self.chatcore_evaluator,
        ) = results

    def _setup_optimizer_and_checkpoint(self):
        """
        Initialize optimizer with scaled learning rates and weight decay.

        Key design decisions:
        1. Batch size scaling: Use sqrt rule (LR ∝ sqrt(batch_size)) because both AdamW and Muon
           are approximately second-order optimizers that benefit from gentler scaling than linear.
        2. Weight decay scaling: Deeper models need less regularization (WD ∝ 1/depth²) because
           they have more capacity and the signal propagates through more layers.
        3. Per-parameter-group learning rates: Different param types have different optimal LRs
           (embeddings learn fast, output head learns slow, etc).
        """
        # Scale learning rates based on batch size (tuned at 2^19 = 524288 tokens)
        # Why sqrt? Second-order optimizers accumulate curvature info, so larger batches
        # give more accurate gradient estimates without needing proportionally higher LR
        batch_lr_scale = 1.0
        reference_batch_size = 2**19
        batch_ratio = self.total_batch_size / reference_batch_size
        if batch_ratio != 1.0:
            batch_lr_scale = batch_ratio**0.5
            if self.master_process:
                print("\n📐 BATCH SIZE SCALING")
                print(
                    f"   Batch ratio: {batch_ratio:.4f} ({self.total_batch_size:,} / {reference_batch_size:,})"
                )
                print(f"   LR scale: {batch_lr_scale:.4f} (sqrt rule)")

        # Scale weight decay by depth (tuned at depth=12)
        # Deeper networks have implicit regularization from depth, so need less explicit L2
        weight_decay = self.config.weight_decay
        if self.config._depth_mode:
            reference_depth = 12
            wd_scale = (reference_depth / self.config.depth) ** 2
            weight_decay = self.config.weight_decay * wd_scale

            if self.master_process:
                print("\n📐 WEIGHT DECAY SCALING")
                print(f"   Depth: {self.config.depth} (reference: {reference_depth})")
                print(
                    f"   WD scale: {wd_scale:.6f} (∝ ({reference_depth}/{self.config.depth})²)"
                )
                print(
                    f"   Weight decay: {self.config.weight_decay:.6f} → {weight_decay:.6f}"
                )

        # Apply batch scaling to all parameter group learning rates
        embedding_lr = self.config.embedding_lr * batch_lr_scale
        unembedding_lr = self.config.unembedding_lr * batch_lr_scale
        matrix_lr = self.config.matrix_lr * batch_lr_scale
        scalar_lr = self.config.scalar_lr * batch_lr_scale

        # Create optimizer(s) - may return single optimizer or list [AdamW, Muon]
        optimizer_result = self.raw_model.configure_optimizers(
            weight_decay=weight_decay,
            device=self.device,
            ddp=self.ddp,
            master_process=self.master_process,
            embedding_lr=embedding_lr,
            unembedding_lr=unembedding_lr,
            matrix_lr=matrix_lr,
            scalar_lr=scalar_lr,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
        )

        # Store scaled weight decay for dynamic scheduling during training
        self.weight_decay_scaled = weight_decay

        # Normalize optimizer result to always be a list for consistent handling
        self.optimizers = (
            optimizer_result
            if isinstance(optimizer_result, list)
            else [optimizer_result]
        )
        self.optimizer = self.optimizers[0]  # Backward compatibility

        # Load checkpoint if provided (three scenarios: resume pretrain, resume SFT, rollover pretrain→SFT)
        if self.checkpoint_path:
            self._load_checkpoint_and_set_counters()

    def _load_checkpoint_and_set_counters(self):
        """
        Load checkpoint and configure training state based on scenario.

        Three checkpoint scenarios:
        1. Resume pretraining: Load weights + optimizer + training counters
        2. Resume SFT: Load weights + optimizer + training counters
        3. Rollover pretrain→SFT: Load weights only, fresh optimizer, reset counters to 0
           (This is transfer learning: pretrained weights, new task, new optimization trajectory)
        """
        # Detect checkpoint type from path (brittle but simple)
        is_pretrain_ckpt = "pretrain_checkpoints" in self.checkpoint_path
        is_sft_ckpt = "sft_checkpoints" in self.checkpoint_path

        # Determine scenario
        is_rollover = self.sft_training and is_pretrain_ckpt  # Transfer learning
        is_resume_pretrain = not self.sft_training and is_pretrain_ckpt
        is_resume_sft = self.sft_training and is_sft_ckpt

        # Validate scenario is recognized
        if not (is_rollover or is_resume_pretrain or is_resume_sft):
            raise ValueError(
                f"Unrecognized checkpoint scenario!\n"
                f"  Checkpoint path: {self.checkpoint_path}\n"
                f"  sft_training flag: {self.sft_training}\n"
                f"Expected patterns:\n"
                f"  - Rollover pretrain→SFT: sft_training=True + 'pretrain_checkpoints' in path\n"
                f"  - Resume pretraining: sft_training=False + 'pretrain_checkpoints' in path\n"
                f"  - Resume SFT: sft_training=True + 'sft_checkpoints' in path"
            )

        # Load checkpoint (weights always, optimizer only when resuming)
        checkpoint_result = load_checkpoint(
            checkpoint_path=self.checkpoint_path,
            model=self.raw_model,
            device=self.device,
            optimizer=(
                self.optimizers if not is_rollover else None
            ),  # Fresh optimizer for rollover
            master_process=self.master_process,
            print_resume_info=not is_rollover,  # Don't print for rollover (we print custom message)
        )

        if checkpoint_result["config"]:
            self.config = checkpoint_result["config"]

        # Set training counters based on scenario
        if is_rollover:
            # Rollover: Start fresh (transfer learning to new task)
            self.start_epoch = 0
            self.start_step = 0
            self.start_global_step = 0
            if self.master_process:
                print(
                    "🔄 Rollover: Pretraining → SFT (weights loaded, fresh optimizer, counters reset)"
                )
                print("   Training will start from global_step: 0")
                print(f"{'='*80}\n")
        else:
            # Resume: Continue from checkpoint counters
            self.start_epoch = checkpoint_result["start_epoch"]
            self.start_step = checkpoint_result["start_step"]
            self.start_global_step = checkpoint_result["start_global_step"]

            if self.master_process:
                mode = "pretraining" if is_resume_pretrain else "SFT"
                print(
                    f"🔄 Resuming {mode} from epoch {self.start_epoch}, "
                    f"step {self.start_step}, global_step {self.start_global_step}"
                )

        # Handle epoch boundary (if checkpoint was saved at end of epoch)
        if self.start_step >= self.max_steps:
            self.start_step = 0
            self.start_epoch += 1

    def _setup_wandb(self):
        """Initialize Weights & Biases logging for experiment tracking."""
        setup_wandb(
            master_process=self.master_process,
            sft_training=self.sft_training,
            config=self.config,
            max_steps=self.max_steps,
            num_epochs=self.num_epochs,
            run_evals=self.run_evals,
            run_core_evals=self.run_core_evals,
            run_chatcore_evals=self.run_chatcore_evals,
            start_step=self.start_step,
            flops_per_token=self.flops_per_token,
            total_batch_size=self.total_batch_size,
        )

    def train(self):
        """
        Main training loop implementing modern LLM training best practices.

        Key components:
        - Gradient accumulation: Simulate large batch sizes with limited memory
        - Mixed precision (bfloat16): 2x speedup with minimal accuracy loss
        - Gradient clipping: Prevent instability from outlier gradients
        - LR scheduling: Warmup prevents early training instability, warmdown helps convergence
        - Periodic evaluation: Track val loss and benchmark performance during training
        """
        training_start_time = time.time()

        # Use TF32 for matmul on Ampere+ GPUs (free 8x speedup over FP32, bit-identical to FP32)
        torch.set_float32_matmul_precision("high")

        # Total optimization steps = steps_per_epoch × num_epochs
        total_steps = self.max_steps * self.num_epochs

        if self.master_process:
            # Calculate FLOPs statistics (using pre-computed values from init)
            total_tokens = self.total_batch_size * total_steps
            total_flops = self.flops_per_token * total_tokens
            num_params = self.raw_model.num_scaling_params()
            tokens_params_ratio = total_tokens / num_params

            print("\n" + "=" * 80)
            print("🚀 STARTING TRAINING")
            print("=" * 80)
            if self.start_global_step > 0:
                print(
                    f"📍 Resuming from epoch {self.start_epoch}, step {self.start_step} "
                    f"(global step {self.start_global_step:,})"
                )
            print(f"📊 Steps per epoch: {self.max_steps:,}")
            print(f"📊 Total epochs: {self.num_epochs}")
            print(f"📊 Total steps: {total_steps:,}")
            print(
                f"📦 Batch size: {self.config.batch_size} x {self.config.block_size} tokens"
            )
            print(f"🌐 World size: {self.ddp_world_size} GPUs")
            print(f"🎯 Total batch size: {self.total_batch_size:,} tokens/step")
            print(f"🔢 Model parameters: {num_params:,}")
            print(f"💫 FLOPs per token: {self.flops_per_token:.3e}")
            print(f"📈 Total training tokens: {total_tokens:,}")
            print(f"⚡ Total training FLOPs: {total_flops:.3e}")
            print(f"📐 Tokens:Params ratio: {tokens_params_ratio:.2f}")

            # Show evaluation schedule
            if self.run_evals:
                num_val_evals = (
                    total_steps // self.run_evals_after
                    if self.run_evals_after > 0
                    else 0
                )
                val_type = (
                    "manual" if self.eval_interval_override is not None else "adaptive"
                )
                print(
                    f"📊 Val loss evals: every {self.run_evals_after} steps → ~{num_val_evals} total ({val_type})"
                )
            if self.run_core_evals:
                num_core_evals = (
                    total_steps // self.run_core_evals_after
                    if self.run_core_evals_after > 0
                    else 0
                )
                core_type = (
                    "manual"
                    if self.core_eval_interval_override is not None
                    else "adaptive"
                )
                print(
                    f"📊 Core benchmark evals: every {self.run_core_evals_after} steps → ~{num_core_evals} total ({core_type})"
                )

            print("=" * 80 + "\n")

        # Main training loop - iterate over epochs and steps
        global_step = (
            self.start_global_step
        )  # Continuous counter across all epochs (for LR schedule)

        for epoch in range(self.start_epoch, self.num_epochs):
            # Resume from mid-epoch if checkpoint was saved there, else start at 0
            epoch_start_step = self.start_step if epoch == self.start_epoch else 0

            for step in range(epoch_start_step, self.max_steps):
                start_time = time.time()

                # Zero out gradients from previous step
                for opt in self.optimizers:
                    opt.zero_grad()

                loss_accumulator = torch.tensor(0.0, device=self.device)
                num_active_tokens = torch.tensor(
                    0, dtype=torch.int64, device=self.device
                )

                # Gradient accumulation: simulate larger batch by accumulating grads over multiple micro-batches
                # This lets us use batch_size=8M tokens even if GPU only fits 512K tokens at once
                for micro_step in range(self.grad_accumulation_steps):
                    # Fetch next batch (dataloader wraps around infinitely)
                    x, y = next(self.train_dataloader)
                    x, y = x.to(self.device), y.to(self.device)

                    # Forward pass in mixed precision (bfloat16 = 2x faster, ~same accuracy as fp32)
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        logits, per_token_loss = self.model(x, y, loss_reduction="none")

                    # Scale loss by accumulation steps so final gradient has correct magnitude
                    # (We're averaging gradients across micro-batches, not summing them)
                    loss = per_token_loss.mean() / self.grad_accumulation_steps
                    loss_accumulator += loss

                    # Track how many tokens we're actually training on
                    # (SFT masks out context with y=-100, pretrain uses all tokens)
                    if self.sft_training:
                        num_active_tokens += (y >= 0).sum()

                    # Backward pass: accumulate gradients (don't step optimizer yet)
                    loss.backward()

                # Synchronize loss and token counts across GPUs (DDP)
                if self.ddp:
                    # Average loss across ranks (each rank computed loss on different data shard)
                    torch.distributed.all_reduce(
                        loss_accumulator, op=torch.distributed.ReduceOp.AVG
                    )
                    # Sum token counts (for throughput calculation)
                    if self.sft_training:
                        torch.distributed.all_reduce(
                            num_active_tokens, op=torch.distributed.ReduceOp.SUM
                        )

                # Clip gradients to prevent training instability from outliers
                # Common in transformers due to attention mechanism and deep architecture
                # Norm is useful for monitoring training health (sudden spikes = trouble)
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )

                # Calculate learning rate multiplier for current step
                # Schedule: linear warmup → constant → cosine warmdown to final_lr_frac
                # Warmup prevents early instability when weights are random
                # Warmdown improves final convergence
                from gpt_2.utils import get_lr_multiplier

                training_phase = "sft" if self.sft_training else "pretrain"
                lrm = get_lr_multiplier(
                    global_step=global_step,
                    max_steps=self.max_steps,
                    num_epochs=self.num_epochs,
                    training_phase=training_phase,
                    warmup_ratio=self.config.warmup_ratio,
                    warmdown_ratio=self.config.warmdown_ratio,
                    final_lr_frac=self.config.final_lr_frac,
                )

                # Get dynamic Muon hyperparameters (momentum and weight decay change during training)
                muon_momentum = get_muon_momentum(global_step)
                muon_weight_decay = get_muon_weight_decay(
                    global_step,
                    self.max_steps,
                    self.num_epochs,
                    self.weight_decay_scaled,
                )

                # Update learning rates for all parameter groups
                # Different param types have different base LRs (stored as initial_lr)
                # All get multiplied by same schedule (lrm)
                for opt in self.optimizers:
                    for param_group in opt.param_groups:
                        param_group["lr"] = param_group["initial_lr"] * lrm

                        # Update Muon-specific hyperparameters (momentum, weight_decay)
                        if param_group.get("kind") == "muon":
                            param_group["momentum"] = muon_momentum
                            param_group["weight_decay"] = muon_weight_decay

                lr = lrm  # Save for logging

                # Take optimizer step (update model parameters using accumulated gradients)
                for opt in self.optimizers:
                    opt.step()

                # Wait for GPU to finish (needed for accurate timing)
                if self.device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()

                # Calculate throughput (tokens/second processed across all GPUs)
                # This is a key metric for training efficiency
                if self.sft_training:
                    # SFT: only count tokens we actually trained on (y >= 0)
                    # Conversation data has context masked out, so counting all tokens would be misleading
                    tokens_per_second = num_active_tokens.item() / (
                        end_time - start_time
                    )
                else:
                    # Pretraining: every token in batch is trained on
                    tokens_per_second = (
                        self.config.batch_size
                        * self.config.block_size
                        * self.grad_accumulation_steps
                        * self.ddp_world_size
                        / (end_time - start_time)
                    )

                # Calculate FLOPs metrics (useful for comparing efficiency across setups)
                flops_per_second = self.flops_per_token * tokens_per_second
                flops_so_far = (
                    self.flops_per_token * self.total_batch_size * global_step
                )

                # MFU = Model FLOPs Utilization (what % of GPU's theoretical peak are we using?)
                # Good MFU is ~50-60% for transformers (attention is memory-bound, not compute-bound)
                mfu = 100 * flops_per_second / (self.peak_flops * self.ddp_world_size)

                # Run validation loss evaluation periodically
                # Val loss is the key metric - tells us if model is learning generalizable patterns
                # If val loss decreases but train loss plateaus = underfitting (need bigger model or more data)
                # If train loss decreases but val loss increases = overfitting (need regularization or less data)
                val_loss = None
                should_eval_val = (
                    global_step % self.run_evals_after == 0
                    or global_step == total_steps - 1
                )
                if self.run_evals and should_eval_val:
                    val_loss = self.evaluator.estimate_validation_loss(
                        step=step, global_step=global_step, total_flops=flops_so_far
                    )

                    # Sample text from model to qualitatively assess what it's learning
                    # Loss numbers are useful but seeing actual generations gives intuition
                    if self.config.enable_sampling and self.master_process:
                        print(f"\n{'='*80}")
                        print(f"📝 SAMPLING FROM MODEL (Step {global_step})")
                        print(f"{'='*80}\n")

                        for i, context in enumerate(self.sample_contexts, 1):
                            print(
                                f"Context {i}/{len(self.sample_contexts)}: {context[:50]}..."
                            )
                            self.evaluator.sample_from_model(
                                num_sequences=1,
                                max_length=self.config.generation_max_length,
                                context=context,
                                step=step,
                            )
                        print(f"{'='*80}\n")

                # Run CORE benchmark evaluations (separate interval, usually less frequent than val loss)
                # These measure actual downstream task performance (not just perplexity)
                should_eval_core = (
                    global_step % self.run_core_evals_after == 0
                    or global_step == total_steps - 1
                )
                if self.run_core_evals and should_eval_core:
                    self.core_evaluator.evaluate_all_tasks(
                        step=step, global_step=global_step
                    )

                # Save checkpoint periodically (for resuming training or rollover to SFT)
                # Save raw_model not DDP-wrapped model (cleaner state_dict without module. prefix)
                checkpoint_interval = (
                    self.config.checkpoint_interval_sft
                    if self.sft_training
                    else self.config.checkpoint_interval_pretrain
                )
                save_checkpoint(
                    model=self.raw_model,
                    optimizer=self.optimizers,
                    step=step,
                    epoch=epoch,
                    global_step=global_step,
                    val_loss=val_loss,
                    checkpoint_dir=self.checkpoint_dir,
                    ddp=self.ddp,
                    checkpoint_interval=checkpoint_interval,
                    max_steps=self.max_steps,
                    num_epochs=self.num_epochs,
                    master_process=self.master_process,
                    sft_training=self.sft_training,
                    depth=self.config.depth,
                )

                # Log metrics to WandB (only master process logs to avoid duplicates)
                if self.master_process:
                    train_loss = loss_accumulator.item()

                    # Build metrics dictionary for logging
                    log_dict = {
                        "step": global_step,
                        "train_loss": train_loss,
                        "lr_multiplier": lr,
                        "tokens_per_second": tokens_per_second,
                        "time_taken": end_time - start_time,
                        "gradient_norm": norm,
                        "flops_per_second": flops_per_second,
                        "total_training_flops_train": flops_so_far,
                        "mfu": mfu,
                    }

                    # Log per-parameter-group learning rates
                    # We use different LRs for different parameter types because they learn at different rates:
                    # - Embeddings (0.3): Fast learning - token representations need to adjust quickly
                    # - Unembedding (0.004): Slow learning - output distribution should be stable
                    # - Matrices (0.02): Medium learning - core transformer weights
                    # - Scalars/Biases (0.5): Fast learning - few parameters, can move quickly
                    opt = self.optimizers[0]
                    adamw_groups = [
                        g for g in opt.param_groups if g.get("kind") == "adamw"
                    ]
                    muon_groups = [
                        g for g in opt.param_groups if g.get("kind") == "muon"
                    ]

                    if len(adamw_groups) >= 3:
                        log_dict["lr/unembedding"] = adamw_groups[0]["lr"]
                        log_dict["lr/embedding"] = adamw_groups[1]["lr"]
                        log_dict["lr/scalar"] = adamw_groups[2]["lr"]

                    if len(muon_groups) > 0:
                        log_dict["lr/matrix"] = muon_groups[0]["lr"]
                        if "momentum" in muon_groups[0]:
                            log_dict["muon/momentum"] = muon_groups[0]["momentum"]
                        if "weight_decay" in muon_groups[0]:
                            log_dict["muon/weight_decay"] = muon_groups[0][
                                "weight_decay"
                            ]

                    wandb.log(log_dict)

                    # Print training progress (helps with monitoring long runs)
                    progress = (global_step + 1) / total_steps * 100
                    print(
                        f"[Epoch {epoch+1}/{self.num_epochs}] [Step {step:>5}/{self.max_steps}] ({progress:>5.1f}%) | "
                        f"Loss: {train_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Grad: {norm:.2e} | "
                        f"Speed: {tokens_per_second/1000:.1f}K tok/s | "
                        f"MFU: {mfu:.2f}% | "
                        f"Time: {end_time - start_time:.2f}s"
                    )

                global_step += 1

            # Run ChatCORE evaluation after each epoch
            if self.run_chatcore_evals:
                if self.master_process:
                    print("\n" + "=" * 80)
                    print(
                        f"🎯 Running ChatCORE evaluation after Epoch {epoch+1}/{self.num_epochs}..."
                    )
                    print("=" * 80 + "\n")

                chatcore_results = self.chatcore_evaluator.evaluate_all_tasks(
                    step=step, global_step=global_step
                )

                if self.master_process:
                    print("\n" + "=" * 80)
                    print(f"📊 ChatCORE RESULTS (Epoch {epoch+1}/{self.num_epochs}):")
                    for task_name, results in chatcore_results.items():
                        accuracy = results["accuracy"]
                        correct = results["correct"]
                        total = results["total"]
                        print(f"   {task_name}: {accuracy:.2%} ({correct}/{total})")
                    print("=" * 80 + "\n")

        # ===== Training complete - cleanup and final reporting =====
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time

        # Print and log dataloader packing stats (master only, aggregated across all ranks)
        # High crop % is fine for pretraining (we see all tokens eventually)
        # High crop % is concerning for SFT (we're wasting instruction-following examples)
        if hasattr(self.train_dataloader, "get_stats"):
            try:
                # Get per-rank stats (already aggregated across dataloader workers)
                stats = self.train_dataloader.get_stats()

                # Aggregate stats across all GPU ranks using torch.distributed.all_reduce
                if self.ddp:
                    # Pack stats into tensor for all_reduce (SUM operation)
                    # Order matters - must unpack in same order
                    stats_tensor = torch.tensor(
                        [
                            stats["total_tokens"],
                            stats["processed_tokens"],
                            stats["cropped_tokens"],
                            stats["corrupted_docs"],
                            stats["empty_docs"],
                        ],
                        dtype=torch.long,
                        device=self.device,
                    )

                    # Sum stats from all ranks
                    torch.distributed.all_reduce(
                        stats_tensor, op=torch.distributed.ReduceOp.SUM
                    )

                    # Unpack aggregated stats back to dict
                    stats["total_tokens"] = stats_tensor[0].item()
                    stats["processed_tokens"] = stats_tensor[1].item()
                    stats["cropped_tokens"] = stats_tensor[2].item()
                    stats["corrupted_docs"] = stats_tensor[3].item()
                    stats["empty_docs"] = stats_tensor[4].item()

                    # Recalculate percentages with global totals
                    stats["cropped_tokens_pct"] = (
                        100.0 * stats["cropped_tokens"] / max(1, stats["total_tokens"])
                    )

                # Only master prints and logs (all ranks now have same stats after all_reduce)
                if self.master_process:
                    rank_label = (
                        f"All {self.ddp_world_size} Ranks"
                        if self.ddp
                        else "Single Process"
                    )
                    print(f"\n{'='*80}")
                    print(f"📦 DATALOADER PACKING STATS (Final - {rank_label})")
                    print(f"{'='*80}")
                    print(f"   Total tokens received: {stats['total_tokens']:,}")
                    print(
                        f"   Processed tokens (packed): {stats['processed_tokens']:,}"
                    )
                    print(
                        f"   Cropped tokens: {stats['cropped_tokens']:,} ({stats['cropped_tokens_pct']:.2f}%)"
                    )
                    print(f"   Corrupted docs: {stats['corrupted_docs']:,}")
                    print(f"   Empty docs: {stats['empty_docs']:,}")
                    print(f"{'='*80}\n")

                    # Log dataloader stats to wandb
                    wandb.log(
                        {
                            "dataloader/total_tokens": stats["total_tokens"],
                            "dataloader/processed_tokens": stats["processed_tokens"],
                            "dataloader/cropped_tokens": stats["cropped_tokens"],
                            "dataloader/cropped_tokens_pct": stats[
                                "cropped_tokens_pct"
                            ],
                            "dataloader/corrupted_docs": stats["corrupted_docs"],
                            "dataloader/empty_docs": stats["empty_docs"],
                        }
                    )
            except Exception as e:
                if self.master_process:
                    print(f"⚠️  Could not retrieve dataloader stats: {e}")

        if self.master_process:
            # Print final training summary
            hours = int(total_training_time // 3600)
            minutes = int((total_training_time % 3600) // 60)
            seconds = int(total_training_time % 60)

            if hours > 0:
                time_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"

            print("\n" + "=" * 80)
            print("✅ TRAINING COMPLETE!")
            print("=" * 80)
            print(f"⏱️  Total training time: {time_str} ({total_training_time:.2f}s)")
            print(f"📊 Total steps completed: {global_step:,}")
            print(f"📊 Average time per step: {total_training_time/global_step:.2f}s")
            print("=" * 80 + "\n")

            # Close wandb (master only)
            wandb.finish()

        # Gracefully shutdown dataloaders (ALL processes must do this!)
        # Each GPU rank has its own dataloader with worker processes that need cleanup
        self._cleanup_dataloaders()

    def _cleanup_dataloaders(self):
        """Gracefully shutdown dataloader workers."""
        # Stop train dataloader workers
        if hasattr(self.train_dataloader, "cleanup"):
            self.train_dataloader.cleanup()
            time.sleep(0.5)  # Give workers time to shutdown

        # Stop eval dataloader workers
        if (
            self.evaluator is not None
            and hasattr(self.evaluator, "eval_dataloader")
            and self.evaluator.eval_dataloader is not None
            and hasattr(self.evaluator.eval_dataloader, "cleanup")
        ):
            self.evaluator.eval_dataloader.cleanup()
            time.sleep(0.5)
