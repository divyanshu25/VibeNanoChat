# Add gpt_2 to python path
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import math
import torch
from gpt_2.gpt2_model import GPT, GPTConfig

# from gpt_2.open_webtext_dataloader import OpenWebtextDataloader
from gpt_2.fineweb_edu_dataloader import FinewebEduDataloader
from gpt_2.hellaswag_dataloader import HellaSwagDataloader
from gpt_2.task_mixture_dataloader import TaskMixtureDataloader
from gpt_2.utils import get_lr, load_checkpoint, save_checkpoint, accumulate_bpb
import time
import wandb
from gpt_2.evaluator import Evaluators
from datetime import datetime


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
        mid_training=False,
        checkpoint_path=None,
        token_bytes_path=None,
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
            mid_training: Whether to do mid-training (uses TaskMixture instead of pretraining data)
            checkpoint_path: Path to checkpoint to load (for mid-training or resuming)
            token_bytes_path: Path to pre-computed token_bytes.pt for BPB calculation
        """
        # Store basic config
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.device = device
        self.master_process = master_process
        self.run_evals = run_evals
        self.mid_training = mid_training
        self.checkpoint_path = checkpoint_path
        self.token_bytes_path = token_bytes_path

        # Initialize start states
        self.start_step = 0
        self.start_epoch = 0
        self.start_global_step = 0

        # Setup components
        self._setup_logging()
        self._setup_model()
        self._setup_hyperparameters()
        self._setup_dataloaders()
        self._setup_optimizer_and_checkpoint()
        self._setup_wandb()
        # self._setup_token_bytes()

    def _setup_logging(self):
        """Setup generation log file for tracking model outputs."""
        if self.master_process:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.generation_log_file = os.path.join(
                log_dir, f"generations_{timestamp}.txt"
            )
            print(f"📝 Saving generations to: {self.generation_log_file}")
        else:
            self.generation_log_file = None

    def _setup_model(self):
        """Initialize GPT model and wrap with DDP if needed."""
        self.config = GPTConfig()
        self.model = GPT(self.config)
        self.model.to(self.device)

        # Optional: Compile model for faster training
        # self.model = torch.compile(self.model)

        if self.ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.ddp_local_rank]
            )
            self.raw_model = self.model.module
        else:
            self.raw_model = self.model

    def _setup_hyperparameters(self):
        """Configure training hyperparameters based on training mode."""
        self.num_epochs = 6
        self.run_evals_after = self.config.eval_interval

        # Batch size and gradient accumulation
        self.total_batch_size = self.config.total_batch_size
        self.grad_accumulation_steps = self.total_batch_size // (
            self.config.batch_size * self.config.block_size * self.ddp_world_size
        )
        assert (
            self.total_batch_size
            % (self.config.batch_size * self.config.block_size * self.ddp_world_size)
            == 0
        ), "Total batch size must be divisible by batch size * block size * world size"

        if self.master_process:
            print(f"Total batch size: {self.total_batch_size}")
            print(f"Grad accumulation steps: {self.grad_accumulation_steps}")

        # Learning rate scheduling parameters
        if self.mid_training:
            self.max_learning_rate = 1e-4
            self.min_learning_rate = self.max_learning_rate * 0.1
            self.warmup_steps = 80
            self.max_steps = 878
        else:
            self.max_learning_rate = 6e-4
            self.min_learning_rate = self.max_learning_rate * 0.1
            self.warmup_steps = 715
            self.max_steps = 18977

    def _setup_dataloaders(self):
        """Initialize train and eval dataloaders based on training mode."""
        # Select dataloader class and data directory based on training mode
        if self.mid_training:
            DataloaderClass = TaskMixtureDataloader
            data_dir = "/sensei-fs/users/divgoyal/nanochat_midtraining_data"
            if self.master_process:
                print("\n" + "=" * 80)
                print("🔄 MID-TRAINING MODE: Using TaskMixture datasets")
                print("=" * 80 + "\n")
        else:
            DataloaderClass = FinewebEduDataloader
            data_dir = "/sensei-fs/users/divgoyal/fineweb_edu"

        self.train_dataloader = DataloaderClass(
            data_dir=data_dir,
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
            ddp_world_size=self.ddp_world_size,
            ddp_rank=self.ddp_rank,
            split="train",
            master_process=self.master_process,
        )

        if self.run_evals:
            self.eval_dataloader = DataloaderClass(
                data_dir=data_dir,
                batch_size=self.config.batch_size,
                block_size=self.config.block_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                split="val",
                master_process=self.master_process,
            )
            self.hellaswag_dataloader = HellaSwagDataloader(
                data_dir="/sensei-fs/users/divgoyal/hellaswag",
                batch_size=self.config.batch_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                split="validation",
                master_process=self.master_process,
            )
            self.evaluator = Evaluators(
                model=self.model,
                eval_dataloader=self.eval_dataloader,
                hellaswag_dataloader=self.hellaswag_dataloader,
                device=self.device,
                master_process=self.master_process,
                ddp=self.ddp,
                ddp_rank=self.ddp_rank,
                ddp_world_size=self.ddp_world_size,
                generation_log_file=self.generation_log_file,
                token_bytes_path=self.token_bytes_path,
            )
        else:
            if self.master_process:
                print("Evaluations disabled - skipping eval dataloader initialization")

    def _setup_optimizer_and_checkpoint(self):
        """Initialize optimizer and load checkpoint if provided."""
        self.optimizer = self.raw_model.configure_optimizers(
            learning_rate=self.max_learning_rate, weight_decay=0.10, device=self.device
        )

        if self.checkpoint_path:
            # For mid-training: only load model weights (fresh optimizer)
            # For resuming pretraining: load both model and optimizer state
            checkpoint_result = load_checkpoint(
                checkpoint_path=self.checkpoint_path,
                model=self.raw_model,
                device=self.device,
                optimizer=None if self.mid_training else self.optimizer,
                master_process=self.master_process,
            )
            if checkpoint_result["config"]:
                self.config = checkpoint_result["config"]

            if self.mid_training:
                self.start_epoch = 0
                self.start_step = 0
                self.start_global_step = 0
                self.config.checkpoint_interval = 800
                if self.master_process:
                    print(
                        "🔄 Mid-training mode: Starting from step 0 (weights loaded, fresh optimizer)"
                    )
            else:
                self.start_epoch = checkpoint_result["start_epoch"]
                self.start_step = checkpoint_result["start_step"]
                self.start_global_step = checkpoint_result["start_global_step"]

            # Handle epoch boundary for resumed checkpoints
            if self.start_step >= self.max_steps:
                self.start_step = 0
                self.start_epoch += 1

    def _setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        if self.master_process:
            project_name = (
                "gpt2-midtraining" if self.mid_training else "gpt2-pretraining"
            )
            wandb.init(
                project=project_name,
                config={
                    "model_type": "GPT-2",
                    "training_mode": (
                        "mid-training" if self.mid_training else "pretraining"
                    ),
                    "batch_size": self.config.batch_size,
                    "block_size": self.config.block_size,
                    "max_learning_rate": self.max_learning_rate,
                    "min_learning_rate": self.min_learning_rate,
                    "warmup_steps": self.warmup_steps,
                    "max_steps": self.max_steps,
                    "num_epochs": self.num_epochs,
                    "weight_decay": 0.10,
                    "gradient_clip_norm": 1.0,
                    "run_evals": self.run_evals,
                    "start_step": self.start_step,
                },
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
        # Set precision for matrix multiplications (improves performance on modern GPUs)
        torch.set_float32_matmul_precision("high")

        if self.master_process:
            total_steps = self.max_steps * self.num_epochs
            print("\n" + "=" * 80)
            if self.mid_training:
                print("🔄 STARTING MID-TRAINING")
            else:
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
            print("=" * 80 + "\n")

        # Main training loop over epochs
        global_step = self.start_global_step  # Track global step across all epochs
        for epoch in range(self.start_epoch, self.num_epochs):
            # Process all batches in the current epoch
            # Only use start_step for the first resumed epoch, then start from 0
            epoch_start_step = self.start_step if epoch == self.start_epoch else 0
            for step in range(epoch_start_step, self.max_steps):
                start_time = time.time()  # Track step timing
                self.optimizer.zero_grad()
                loss_accumulator = torch.tensor(0.0, device=self.device)
                # BPB accumulators
                total_nats = torch.tensor(0.0, dtype=torch.float32, device=self.device)
                total_bytes = torch.tensor(0, dtype=torch.int64, device=self.device)

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

                    # Accumulate for BPB calculation
                    # if self._token_bytes is not None:
                    #     nats, bytes = accumulate_bpb(per_token_loss, y, self._token_bytes)
                    #     total_nats += nats
                    #     total_bytes += bytes

                    if self.ddp:
                        self.model.require_backward_grad_sync = (
                            micro_step == self.grad_accumulation_steps - 1
                        )
                    # Backward pass: compute gradients
                    loss.backward()
                if self.ddp:
                    torch.distributed.all_reduce(
                        loss_accumulator, op=torch.distributed.ReduceOp.AVG
                    )
                    # if self._token_bytes is not None:
                    #     torch.distributed.all_reduce(
                    #         total_nats, op=torch.distributed.ReduceOp.SUM
                    #     )
                    #     torch.distributed.all_reduce(
                    #         total_bytes, op=torch.distributed.ReduceOp.SUM
                    #     )

                # Gradient clipping to prevent exploding gradients
                # This stabilizes training by limiting gradient magnitude
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )  # Clip gradients to max norm of 1.0

                # Update learning rate based on global step (continuous across epochs)
                lr = get_lr(
                    global_step=global_step,
                    warmup_steps=self.warmup_steps,
                    max_steps=self.max_steps,
                    num_epochs=self.num_epochs,
                    max_learning_rate=self.max_learning_rate,
                    min_learning_rate=self.min_learning_rate,
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                # Apply gradients to update model parameters
                self.optimizer.step()

                # Synchronize CUDA operations for accurate timing
                if self.device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()

                # Calculate training throughput (tokens processed per second)
                tokens_per_second = (
                    self.train_dataloader.batch_size
                    * self.train_dataloader.block_size
                    * self.grad_accumulation_steps
                    * self.ddp_world_size
                    / (end_time - start_time)
                )

                # Periodically estimate loss on train/val sets for monitoring
                total_steps = self.max_steps * self.num_epochs
                val_loss = None  # Will be set if evals run
                should_eval = (
                    global_step % self.run_evals_after == 0
                    or global_step == total_steps - 1
                )
                if self.run_evals and should_eval:
                    val_loss = self.evaluator.estimate_validation_loss(
                        step=step, global_step=global_step
                    )
                    if not self.mid_training:
                        self.evaluator.estimate_hellaswag_accuracy(
                            step=step, global_step=global_step
                        )
                    sample_context = (
                        "<|bos|><|user_start|>Give me a random joke.<|user_end|>"
                        if self.mid_training
                        else "Hello, I'm a language model,"
                    )
                    self.evaluator.sample_from_model(
                        num_sequences=4,
                        max_length=32,
                        context=sample_context,
                        step=step,
                    )

                # Save checkpoint at intervals or at end of training (independent of evals)
                save_checkpoint(
                    model=self.model,
                    optimizer=self.optimizer,
                    step=step,
                    epoch=epoch,
                    global_step=global_step,
                    val_loss=val_loss,
                    checkpoint_dir="/sensei-fs/users/divgoyal/nanogpt/checkpoints",
                    ddp=self.ddp,
                    checkpoint_interval=self.config.checkpoint_interval,
                    max_steps=self.max_steps,
                    num_epochs=self.num_epochs,
                    master_process=self.master_process,
                    mid_training=self.mid_training,
                )
                # Log metrics to wandb
                if self.master_process:
                    train_loss = loss_accumulator.item()
                    # # Compute train BPB
                    # total_bytes_val = total_bytes.item()
                    # if total_bytes_val > 0:
                    #     train_bpb = total_nats.item() / (math.log(2) * total_bytes_val)
                    # else:
                    #     train_bpb = float("inf")

                    wandb.log(
                        {
                            "epoch": epoch,
                            "epoch_step": step,
                            "step": global_step,
                            "train_loss": train_loss,
                            # "train_bpb": train_bpb,
                            "learning_rate": lr,
                            "tokens_per_second": tokens_per_second,
                            "time_taken": end_time - start_time,
                            "gradient_norm": norm,
                        }
                    )

                    # Print comprehensive training statistics
                    total_steps = self.max_steps * self.num_epochs
                    progress = (global_step + 1) / total_steps * 100
                    print(
                        f"[Epoch {epoch+1}/{self.num_epochs}] [Step {step:>5}/{self.max_steps}] ({progress:>5.1f}%) | "
                        f"Loss: {train_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Grad: {norm:.2e} | "
                        f"Speed: {tokens_per_second/1000:.1f}K tok/s | "
                        f"Time: {end_time - start_time:.2f}s"
                    )

                # Increment global step counter
                global_step += 1

        # Training complete
        if self.master_process:
            print("\n" + "=" * 80)
            print("✅ TRAINING COMPLETE!")
            print("=" * 80 + "\n")

        # Finish wandb run
        wandb.finish()
