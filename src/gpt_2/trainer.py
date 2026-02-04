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
from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT
from gpt_2.muon import get_muon_momentum, get_muon_weight_decay
from gpt_2.sample_contexts import GENERAL_SAMPLE_CONTEXTS, SFT_SAMPLE_CONTEXTS
from gpt_2.training_utilities import (setup_dataloaders, setup_logging,
                                      setup_wandb)
from gpt_2.utils import (calculate_num_iterations, get_peak_flops,
                         load_checkpoint, save_checkpoint)


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
        # Store basic config
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.device = device
        self.master_process = master_process
        self.run_evals = run_evals
        self.run_core_evals = run_core_evals
        self.run_chatcore_evals = run_chatcore_evals
        self.sft_training = sft_training
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = checkpoint_dir
        self.token_bytes_path = token_bytes_path
        self.depth_override = depth
        self.aspect_ratio_override = aspect_ratio
        self.head_dim_override = head_dim
        self.target_flops_override = target_flops
        self.param_data_ratio_override = param_data_ratio
        self.eval_interval_override = eval_interval
        self.core_eval_interval_override = core_eval_interval

        # Initialize start states
        self.start_step = 0
        self.start_epoch = 0
        self.start_global_step = 0

        # Select 4 random sample contexts for generation evaluation
        if self.sft_training:
            self.sample_contexts = random.sample(
                SFT_SAMPLE_CONTEXTS, min(4, len(SFT_SAMPLE_CONTEXTS))
            )
        else:
            self.sample_contexts = random.sample(
                GENERAL_SAMPLE_CONTEXTS, min(4, len(GENERAL_SAMPLE_CONTEXTS))
            )

        # Setup components
        self._setup_model()
        # Setup logging after model so we have depth info for filename
        self.generation_log_file = setup_logging(
            self.master_process,
            depth=self.config.depth if self.config._depth_mode else None,
            sft_training=self.sft_training,
        )
        self._setup_hyperparameters()
        self._setup_dataloaders_wrapper()
        self._setup_optimizer_and_checkpoint()
        self._setup_wandb_wrapper()
        # self._setup_token_bytes()

    def _setup_model(self):
        """Initialize GPT model and wrap with DDP if needed."""
        self.config = GPTConfig()

        # Override architecture parameters using depth-based parameterization
        if self.depth_override is not None:
            if self.master_process:
                print(f"🔢 Using depth-based architecture: depth={self.depth_override}")
            self.config.depth = self.depth_override
            if self.aspect_ratio_override is not None:
                self.config.aspect_ratio = self.aspect_ratio_override
            if self.head_dim_override is not None:
                self.config.head_dim = self.head_dim_override
            # __post_init__ will auto-calculate n_layer, n_embed, n_head
            self.config.__post_init__()

            # Auto-scale batch size based on model size to avoid OOM
            # Small models can use larger batch sizes for faster training
            # Large models need smaller batch sizes to fit in memory
            # Reference: tuned on 2x H100 80GB
            if self.depth_override <= 8:
                auto_batch_size = 32  # ~100M params: high throughput
            elif self.depth_override <= 10:
                auto_batch_size = 32  # ~100M params: high throughput
            elif self.depth_override <= 14:
                auto_batch_size = 16  # ~150M-210M params: balanced
            elif self.depth_override <= 18:
                auto_batch_size = 8  # ~280M-360M params: conservative
            elif self.depth_override <= 22:
                auto_batch_size = 4  # ~560M-700M params: tight fit
            else:
                auto_batch_size = 2  # ~1B+ params: minimal batch

            if self.master_process:
                if auto_batch_size != self.config.batch_size:
                    print(
                        f"   Auto-scaling batch_size: {self.config.batch_size} → {auto_batch_size} (for depth={self.depth_override})"
                    )
                else:
                    print(
                        f"   Batch size: {auto_batch_size} (optimal for depth={self.depth_override})"
                    )
            self.config.batch_size = auto_batch_size
        else:
            # Use default config values
            if self.master_process:
                print(
                    f"📝 Using default architecture: n_layer={self.config.n_layer}, n_embed={self.config.n_embed}"
                )

        # Override target_flops if provided
        if self.target_flops_override is not None:
            if self.master_process:
                print(
                    f"🎯 Overriding target_flops: {self.config.target_flops:.2e} → {self.target_flops_override:.2e}"
                )
            self.config.target_flops = self.target_flops_override

        # Override param_data_ratio if provided
        if self.param_data_ratio_override is not None:
            if self.master_process:
                print(
                    f"🎯 Overriding param_data_ratio: {self.config.target_param_data_ratio} → {self.param_data_ratio_override}"
                )
            self.config.target_param_data_ratio = self.param_data_ratio_override

        # Override eval_interval if provided
        if self.eval_interval_override is not None:
            if self.master_process:
                print(
                    f"⏱️  Overriding eval_interval: {self.config.eval_interval} → {self.eval_interval_override}"
                )
            self.config.eval_interval = self.eval_interval_override

        # Create raw model and keep reference BEFORE any wrapping
        # This reference will always point to the unwrapped model with updated weights
        self.raw_model = GPT(self.config, master_process=self.master_process)
        self.raw_model.to(self.device)

        # Wrap with torch.compile for faster training
        # Use dynamic=False because input shapes never change during training (nanochat-style)
        # Requires PyTorch 2.9.0+ to avoid none_dealloc crash
        self.model = torch.compile(self.raw_model, dynamic=False)
        # self.model = self.raw_model

        # Nanochat-style: NO DDP wrapper when using combined optimizer with Muon
        # The DistMuonAdamW optimizer handles gradient synchronization internally
        # This provides better overlap of communication with computation
        #
        # Nanochat-style: NO DDP wrapper when using Muon optimizer
        # The DistMuonAdamW optimizer handles gradient synchronization internally
        if self.ddp:
            if self.master_process:
                print(
                    "Using DistMuonAdamW optimizer - skipping DDP wrapper (nanochat-style)"
                )

        # Note: self.raw_model stays unchanged and shares parameters with self.model
        # This allows clean checkpoint saving without unwrapping

    def _setup_hyperparameters(self):
        """Configure training hyperparameters based on training mode."""
        self.num_epochs = self.config.num_epochs
        # Note: self.run_evals_after will be set adaptively after we know max_steps

        # Batch size and gradient accumulation
        if self.sft_training:
            # For SFT, we count examples (conversations) not tokens
            # No gradient accumulation - process each batch immediately
            self.grad_accumulation_steps = 1
            self.total_batch_size = self.config.batch_size * self.ddp_world_size
        else:
            # For pretrain/midtrain, we count tokens
            self.total_batch_size = self.config.total_batch_size
            self.grad_accumulation_steps = self.total_batch_size // (
                self.config.batch_size * self.config.block_size * self.ddp_world_size
            )

            assert (
                self.total_batch_size
                % (
                    self.config.batch_size
                    * self.config.block_size
                    * self.ddp_world_size
                )
                == 0
            ), "Total batch size must be divisible by batch_size * block_size * world_size"

        if self.master_process:
            print(f"Total batch size: {self.total_batch_size}")
            print(f"Grad accumulation steps: {self.grad_accumulation_steps}")

        # Learning rate scheduling parameters
        self.max_learning_rate = self.config.max_learning_rate
        self.min_learning_rate = self.max_learning_rate * self.config.min_lr_ratio

        # Apply nanochat-style depth-aware LR scaling if using depth mode
        # LR ∝ 1/√model_dim (tuned at model_dim=768, depth=12)
        if self.config._depth_mode:
            # Learning rate scaling: LR ∝ 1/√model_dim
            # Reference: tuned at model_dim=768 (depth=12, aspect_ratio=64)
            reference_dim = 768
            lr_scale = (self.config.n_embed / reference_dim) ** -0.5
            self.max_learning_rate *= lr_scale
            self.min_learning_rate *= lr_scale

            if self.master_process:
                print(f"\n📐 DEPTH-AWARE SCALING (depth={self.config.depth})")
                print(f"   n_layer: {self.config.n_layer}")
                print(
                    f"   n_embed: {self.config.n_embed} (base: {self.config._base_dim}, nudge: {self.config._nudge:+d})"
                )
                print(f"   n_head: {self.config.n_head}")
                print(f"   head_dim: {self.config.n_embed // self.config.n_head}")
                print(
                    f"   LR scaling: {lr_scale:.6f} (∝ 1/√({self.config.n_embed}/{reference_dim}))"
                )
                print(
                    f"   Max LR: {self.config.max_learning_rate:.2e} → {self.max_learning_rate:.2e}"
                )

        # Automatically calculate steps based on config settings for all phases
        if self.master_process:
            print("\n" + "=" * 80)
            if self.sft_training:
                print("📊 CALCULATING SFT TRAINING STEPS")
            else:
                print("📊 CALCULATING PRETRAINING STEPS")
            print("=" * 80)

        num_iterations, flops_per_token, _ = calculate_num_iterations(
            self.raw_model, self.config, self.master_process
        )
        self.max_steps = num_iterations
        self.flops_per_token = flops_per_token

        if self.master_process:
            print("=" * 80 + "\n")

        # Initialize peak FLOPs for MFU calculation
        if self.device.startswith("cuda"):
            device_name = torch.cuda.get_device_name(self.device)
            self.peak_flops = get_peak_flops(device_name)
            if self.master_process:
                print(f"GPU: {device_name}")
                print(f"Peak FLOPS (BF16): {self.peak_flops:.2e}\n")
        else:
            self.peak_flops = float("inf")  # MFU not meaningful for non-CUDA devices

        # Set warmup steps based on training phase (calculated as % of max_steps)
        if self.sft_training:
            self.warmup_steps = int(self.max_steps * self.config.lr_warmup_ratio_sft)
        else:
            self.warmup_steps = int(
                self.max_steps * self.config.lr_warmup_ratio_pretrain
            )

        # Set adaptive eval intervals based on total training steps
        # Val loss: frequent (good learning curve), Core evals: sparse (expensive benchmarks)
        total_steps = self.max_steps * self.num_epochs

        # Validation loss interval (faster, more frequent)
        if self.eval_interval_override is not None:
            self.run_evals_after = self.eval_interval_override
        else:
            # Adaptive: target ~8-12 val loss measurements for smooth learning curve
            adaptive_val_interval = max(100, total_steps // 10)  # min 100 steps
            adaptive_val_interval = min(
                adaptive_val_interval, 500
            )  # max 500 steps (don't spam)
            self.run_evals_after = adaptive_val_interval

        # Core evaluation interval (slower, less frequent)
        if self.core_eval_interval_override is not None:
            self.run_core_evals_after = self.core_eval_interval_override
        else:
            # Adaptive: target ~3-5 core eval measurements (expensive, 80s each)
            adaptive_core_interval = max(250, total_steps // 4)  # min 250 steps
            adaptive_core_interval = min(
                adaptive_core_interval, total_steps // 2
            )  # at least 2 evals
            self.run_core_evals_after = adaptive_core_interval

    def _setup_dataloaders_wrapper(self):
        """Wrapper to setup dataloaders using utility functions."""
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
        """Initialize optimizer and load checkpoint if provided."""
        # ===== Nanochat-style Batch Size Scaling (sqrt rule) =====
        # Learning rates are tuned at reference batch size 2^19 (524288)
        batch_lr_scale = 1.0
        reference_batch_size = 2**19
        batch_ratio = self.total_batch_size / reference_batch_size
        if batch_ratio != 1.0:
            # Muon: sqrt scaling (second-order-ish optimizer)
            # AdamW: sqrt scaling is standard
            batch_lr_scale = batch_ratio**0.5
            if self.master_process:
                print("\n📐 BATCH SIZE SCALING")
                print(
                    f"   Batch ratio: {batch_ratio:.4f} ({self.total_batch_size:,} / {reference_batch_size:,})"
                )
                print(f"   LR scale: {batch_lr_scale:.4f} (sqrt rule)")

        # ===== Nanochat-style Weight Decay Scaling by Depth =====
        # Weight decay is tuned at depth=12, scales as ∝ 1/depth²
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

        # ===== Apply Batch Size Scaling to Learning Rates =====
        embedding_lr = self.config.embedding_lr * batch_lr_scale
        unembedding_lr = self.config.unembedding_lr * batch_lr_scale
        matrix_lr = self.config.matrix_lr * batch_lr_scale
        scalar_lr = self.config.scalar_lr * batch_lr_scale

        optimizer_result = self.raw_model.configure_optimizers(
            learning_rate=self.max_learning_rate,
            weight_decay=weight_decay,
            device=self.device,
            muon_lr=self.config.muon_lr,
            ddp=self.ddp,
            master_process=self.master_process,
            embedding_lr=embedding_lr,
            unembedding_lr=unembedding_lr,
            matrix_lr=matrix_lr,
            scalar_lr=scalar_lr,
            adam_beta1=self.config.adam_beta1,
            adam_beta2=self.config.adam_beta2,
        )

        # Store scaled weight decay for scheduler
        self.weight_decay_scaled = weight_decay

        # Handle both single optimizer and list of optimizers
        if isinstance(optimizer_result, list):
            self.optimizers = optimizer_result  # List of optimizers (AdamW + Muon)
            self.optimizer = optimizer_result[0]  # Keep for backward compatibility
        else:
            self.optimizer = optimizer_result
            self.optimizers = [optimizer_result]

        if self.checkpoint_path:
            # Determine checkpoint source
            is_pretrain_ckpt = "pretrain_checkpoints" in self.checkpoint_path
            is_sft_ckpt = "sft_checkpoints" in self.checkpoint_path

            # Define training scenario flags
            is_rollover_pretrain_to_sft = self.sft_training and is_pretrain_ckpt
            is_resume_pretrain = not self.sft_training and is_pretrain_ckpt
            is_resume_sft = self.sft_training and is_sft_ckpt

            # Load optimizer only when resuming (not when rolling over)
            should_load_optimizer = not is_rollover_pretrain_to_sft
            # Don't print resume info for rollover scenarios
            should_print_resume_info = not is_rollover_pretrain_to_sft

            checkpoint_result = load_checkpoint(
                checkpoint_path=self.checkpoint_path,
                model=self.raw_model,
                device=self.device,
                optimizer=self.optimizers if should_load_optimizer else None,
                master_process=self.master_process,
                print_resume_info=should_print_resume_info,
            )
            if checkpoint_result["config"]:
                self.config = checkpoint_result["config"]

            # Reset training counters only when rolling over
            if is_rollover_pretrain_to_sft:
                self.start_epoch = 0
                self.start_step = 0
                self.start_global_step = 0
                if self.master_process:
                    print(
                        "🔄 Rollover: Pretraining → SFT "
                        "(weights loaded, fresh optimizer, counters reset to 0)"
                    )
                    print("   Training will start from global_step: 0")
                    print(f"{'='*80}\n")
            # Keep checkpoint counters when resuming
            elif is_resume_pretrain or is_resume_sft:
                self.start_epoch = checkpoint_result["start_epoch"]
                self.start_step = checkpoint_result["start_step"]
                self.start_global_step = checkpoint_result["start_global_step"]

                if self.master_process:
                    mode = "pretraining" if is_resume_pretrain else "SFT"
                    print(
                        f"🔄 Resuming {mode} from epoch {self.start_epoch}, "
                        f"step {self.start_step}, global_step {self.start_global_step} "
                        "(weights + optimizer loaded)"
                    )
            else:
                # Abort if checkpoint scenario is not recognized
                raise ValueError(
                    f"Unrecognized checkpoint scenario!\n"
                    f"  Checkpoint path: {self.checkpoint_path}\n"
                    f"  sft_training flag: {self.sft_training}\n"
                    f"  is_pretrain_ckpt: {is_pretrain_ckpt}\n"
                    f"  is_sft_ckpt: {is_sft_ckpt}\n\n"
                    f"Expected checkpoint path patterns:\n"
                    f"  - For rollover pretrain→sft: sft_training=True + 'pretrain_checkpoints' in path\n"
                    f"  - For resume pretraining: sft_training=False + 'pretrain_checkpoints' in path\n"
                    f"  - For resume SFT: sft_training=True + 'sft_checkpoints' in path"
                )

            # Handle epoch boundary for resumed checkpoints
            if self.start_step >= self.max_steps:
                self.start_step = 0
                self.start_epoch += 1

    def _setup_wandb_wrapper(self):
        """Wrapper to setup wandb using utility function."""
        setup_wandb(
            master_process=self.master_process,
            sft_training=self.sft_training,
            config=self.config,
            max_learning_rate=self.max_learning_rate,
            min_learning_rate=self.min_learning_rate,
            warmup_steps=self.warmup_steps,
            max_steps=self.max_steps,
            num_epochs=self.num_epochs,
            run_evals=self.run_evals,
            run_core_evals=self.run_core_evals,
            run_chatcore_evals=self.run_chatcore_evals,
            start_step=self.start_step,
            flops_per_token=self.flops_per_token,
            total_batch_size=self.total_batch_size,
        )

    def _setup_token_bytes(self):
        """Load pre-computed token_bytes tensor for BPB calculation."""
        if self.token_bytes_path is not None:
            self._token_bytes = torch.load(self.token_bytes_path, weights_only=True).to(
                self.device
            )
        else:
            self._token_bytes = None
            if self.master_process:
                print(
                    "⚠️  token_bytes_path not provided - train BPB will not be computed"
                )

    def train(self):
        """
        Main training loop that implements the full training procedure.
        Includes gradient clipping, learning rate scheduling, and progress monitoring.
        """
        ## Start training ##
        # Record training start time for total duration logging
        training_start_time = time.time()

        # Set precision for matrix multiplications (improves performance on modern GPUs)
        torch.set_float32_matmul_precision("high")

        # Calculate total steps once for the entire training run
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

        # Main training loop over epochs
        global_step = self.start_global_step  # Track global step across all epochs
        for epoch in range(self.start_epoch, self.num_epochs):
            # Process all batches in the current epoch
            # Only use start_step for the first resumed epoch, then start from 0
            epoch_start_step = self.start_step if epoch == self.start_epoch else 0

            # Unified training loop for all modes (pretrain/midtrain/sft)
            for step in range(epoch_start_step, self.max_steps):
                start_time = time.time()  # Track step timing
                # Zero gradients for all optimizers
                for opt in self.optimizers:
                    opt.zero_grad()
                loss_accumulator = torch.tensor(0.0, device=self.device)
                # # BPB accumulators
                # torch.tensor(0.0, dtype=torch.float32, device=self.device)
                # torch.tensor(0, dtype=torch.int64, device=self.device)

                # Track active tokens for SFT training (where targets >= 0)
                num_active_tokens = torch.tensor(
                    0, dtype=torch.int64, device=self.device
                )

                for micro_step in range(self.grad_accumulation_steps):
                    # Get training batch and move to device
                    x, y = self.train_dataloader.next_batch()
                    x, y = x.to(self.device), y.to(self.device)

                    # Forward pass: compute per-token loss
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        logits, per_token_loss = self.model(x, y, loss_reduction="none")

                    # Compute mean loss for backprop
                    loss = per_token_loss.mean() / self.grad_accumulation_steps
                    loss_accumulator += loss

                    # Count active tokens (for SFT: targets >= 0, for pretrain: all tokens)
                    if self.sft_training:
                        num_active_tokens += (y >= 0).sum()
                    loss.backward()

                if self.ddp:
                    torch.distributed.all_reduce(
                        loss_accumulator, op=torch.distributed.ReduceOp.AVG
                    )
                    # Sum active tokens across all ranks for SFT
                    if self.sft_training:
                        torch.distributed.all_reduce(
                            num_active_tokens, op=torch.distributed.ReduceOp.SUM
                        )

                # Gradient clipping to prevent exploding gradients
                # This stabilizes training by limiting gradient magnitude
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )

                # Update learning rate based on global step (continuous across epochs)
                # Nanochat-style: Use get_lr_multiplier to get a multiplier for all param groups
                from gpt_2.utils import get_lr_multiplier

                # Determine training phase
                if self.sft_training:
                    training_phase = "sft"
                else:
                    training_phase = "pretrain"

                lrm = get_lr_multiplier(
                    global_step=global_step,
                    max_steps=self.max_steps,
                    num_epochs=self.num_epochs,
                    training_phase=training_phase,
                    warmup_ratio=self.config.warmup_ratio,
                    warmdown_ratio=self.config.warmdown_ratio,
                    final_lr_frac=self.config.final_lr_frac,
                )

                # Get Muon schedulers
                muon_momentum = get_muon_momentum(global_step)
                muon_weight_decay = get_muon_weight_decay(
                    global_step,
                    self.max_steps,
                    self.num_epochs,
                    self.weight_decay_scaled,
                )

                # Update all optimizers with nanochat-style scheduling
                for opt in self.optimizers:
                    for param_group in opt.param_groups:
                        # Apply LR multiplier to initial_lr
                        param_group["lr"] = param_group["initial_lr"] * lrm

                        # Update Muon-specific params
                        if param_group.get("kind") == "muon":
                            param_group["momentum"] = muon_momentum
                            param_group["weight_decay"] = muon_weight_decay

                lr = lrm  # For logging

                # Apply gradients to update model parameters
                for opt in self.optimizers:
                    opt.step()

                # Synchronize CUDA operations for accurate timing
                if self.device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()

                # Calculate training throughput (tokens processed per second)
                if self.sft_training:
                    # For SFT: count only active tokens (where targets >= 0)
                    tokens_per_second = num_active_tokens.item() / (
                        end_time - start_time
                    )
                else:
                    # For pretrain/midtrain: all tokens in batch are active
                    tokens_per_second = (
                        self.config.batch_size
                        * self.config.block_size
                        * self.grad_accumulation_steps
                        * self.ddp_world_size
                        / (end_time - start_time)
                    )

                # Calculate FLOPs metrics
                flops_per_second = self.flops_per_token * tokens_per_second
                flops_so_far = (
                    self.flops_per_token * self.total_batch_size * global_step
                )
                # MFU: Model FLOPs Utilization (% of theoretical peak performance)
                mfu = 100 * flops_per_second / (self.peak_flops * self.ddp_world_size)

                # Periodically estimate loss on train/val sets for monitoring
                val_loss = None  # Will be set if evals run
                should_eval_val = (
                    global_step % self.run_evals_after == 0
                    or global_step == total_steps - 1
                )
                if self.run_evals and should_eval_val:
                    val_loss = self.evaluator.estimate_validation_loss(
                        step=step, global_step=global_step, total_flops=flops_so_far
                    )

                    # Sample from model to see what it's learning
                    if self.config.enable_sampling and self.master_process:
                        print(f"\n{'='*80}")
                        print(f"📝 SAMPLING FROM MODEL (Step {global_step})")
                        print(f"{'='*80}\n")

                        # Sample from multiple contexts to test different capabilities
                        for i, context in enumerate(self.sample_contexts, 1):
                            print(
                                f"Context {i}/{len(self.sample_contexts)}: {context[:50]}..."
                            )
                            self.evaluator.sample_from_model(
                                num_sequences=1,  # 1 sample per context to avoid spam
                                max_length=self.config.generation_max_length,
                                context=context,
                                step=step,
                            )
                        print(f"{'='*80}\n")

                # Run CORE evaluations if enabled (separate interval from val loss)
                should_eval_core = (
                    global_step % self.run_core_evals_after == 0
                    or global_step == total_steps - 1
                )
                if self.run_core_evals and should_eval_core:
                    self.core_evaluator.evaluate_all_tasks(
                        step=step, global_step=global_step
                    )

                # Save checkpoint at intervals or at end of training (independent of evals)
                if self.sft_training:
                    checkpoint_interval = self.config.checkpoint_interval_sft
                else:
                    checkpoint_interval = self.config.checkpoint_interval_pretrain

                save_checkpoint(
                    model=self.raw_model,  # Pass unwrapped model for clean state_dict
                    optimizer=self.optimizers,  # Pass all optimizers (AdamW + Muon)
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

                # Log metrics to wandb
                if self.master_process:
                    train_loss = loss_accumulator.item()

                    # Build logging dict with base metrics
                    log_dict = {
                        # "epoch": epoch,
                        # "epoch_step": step,
                        "step": global_step,
                        "train_loss": train_loss,
                        # "train_bpb": train_bpb,
                        "lr_multiplier": lr,  # LR schedule multiplier (for reference)
                        "tokens_per_second": tokens_per_second,
                        "time_taken": end_time - start_time,
                        "gradient_norm": norm,
                        "flops_per_second": flops_per_second,
                        "total_training_flops_train": flops_so_far,
                        "mfu": mfu,
                    }

                    # ===== Log Learning Rates =====
                    # We use separate LRs for different parameter types (nanochat-style):
                    # - Embeddings: Fast learning (default 0.3) - adapts token representations quickly
                    # - Unembedding: Slow learning (default 0.004) - stable output distribution
                    # - Matrices: Medium learning (default 0.02) - core transformer weights with Muon
                    # - Scalars/Biases: Fast learning (default 0.5) - small params adjust quickly

                    # Combined optimizer (nanochat-style)
                    opt = self.optimizers[0]

                    # Iterate through param_groups to find specific types
                    adamw_groups = [
                        g for g in opt.param_groups if g.get("kind") == "adamw"
                    ]
                    muon_groups = [
                        g for g in opt.param_groups if g.get("kind") == "muon"
                    ]

                    # Log AdamW learning rates (expect 3 groups: lm_head, embeddings, scalars)
                    if len(adamw_groups) >= 3:
                        log_dict["lr/unembedding"] = adamw_groups[0]["lr"]  # lm_head
                        log_dict["lr/embedding"] = adamw_groups[1]["lr"]  # embeddings
                        log_dict["lr/scalar"] = adamw_groups[2]["lr"]  # scalars/biases

                    # Log Muon learning rate and hyperparameters
                    if len(muon_groups) > 0:
                        log_dict["lr/matrix"] = muon_groups[0][
                            "lr"
                        ]  # All 2D transformer matrices

                        # Log Muon-specific hyperparameters
                        if "momentum" in muon_groups[0]:
                            log_dict["muon/momentum"] = muon_groups[0]["momentum"]
                        if "weight_decay" in muon_groups[0]:
                            log_dict["muon/weight_decay"] = muon_groups[0][
                                "weight_decay"
                            ]

                    wandb.log(log_dict)

                    # Print comprehensive training statistics
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

                # Increment global step counter
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

        # Training complete
        training_end_time = time.time()
        total_training_time = training_end_time - training_start_time

        if self.master_process:
            # Format training time in human-readable format
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

        # Finish wandb run
        wandb.finish()
