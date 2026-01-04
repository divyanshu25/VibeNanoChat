# Add gpt_2 to python path
import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch
from gpt_2.gpt2_model import GPT, GPTConfig
from gpt_2.open_webtext_dataloader import OpenWebtextDataloader
from gpt_2.hellaswag_dataloader import HellaSwagDataloader
from gpt_2.task_mixture_dataloader import TaskMixtureDataloader
import time
from gpt_2.gpt2_model import generate
import math
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
        """
        # Initialize ddp variables
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.ddp = ddp
        self.device = device
        self.master_process = master_process
        self.run_evals = run_evals
        self.mid_training = mid_training
        self.checkpoint_path = checkpoint_path

        # Setup generation log file (only on master process)
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

        # Initialize GPT model with default configuration
        self.config = GPTConfig()
        self.model = GPT(self.config)

        # Load checkpoint if provided (for mid-training or resuming)
        self.start_step = 0
        if self.checkpoint_path:
            self._load_checkpoint(self.checkpoint_path)

        self.model.to(self.device)

        # Optional: Compile model for faster training (commented out to avoid warnings)
        # Use "reduce-overhead" mode instead of "default" to avoid SM warnings on consumer hardware
        # self.model = torch.compile(self.model)

        if self.ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.ddp_local_rank]
            )
            self.raw_model = self.model.module if self.ddp else self.model
        else:
            self.raw_model = self.model

        # Training hyperparameters
        self.num_epochs = 3  # Number of complete passes through the dataset

        self.run_evals_after = self.config.eval_interval  # Run evals every N steps

        self.total_batch_size = self.config.total_batch_size
        self.grad_accumulation_steps = self.total_batch_size // (
            self.config.batch_size * self.config.block_size * self.ddp_world_size
        )  # grad accumulation steps is the total batch size divided by the batch size and block size and ddp world size
        assert (
            self.total_batch_size
            % (self.config.batch_size * self.config.block_size * self.ddp_world_size)
            == 0
        ), "Total batch size must be divisible by batch size and block size and ddp world size"

        # Print total batch size and grad accumulation steps
        if self.master_process:
            print(f"Total batch size: {self.total_batch_size}")
            print(f"Grad accumulation steps: {self.grad_accumulation_steps}")

        # Learning rate scheduling parameters
        if self.mid_training:
            # Lower learning rate for mid-training/fine-tuning
            self.max_learning_rate = 1e-4  # Lower LR for fine-tuning
            self.min_learning_rate = self.max_learning_rate * 0.1
            self.warmup_steps = 100  # Shorter warmup for fine-tuning
            self.max_steps = 5000  # Fewer steps for mid-training
        else:
            # Standard pretraining parameters
            self.max_learning_rate = 6e-4  # Peak learning rate
            self.min_learning_rate = (
                self.max_learning_rate * 0.1
            )  # Minimum learning rate (10% of max)
            self.warmup_steps = 715  # Steps to warm up from 0 to max learning rate
            self.max_steps = 17234  # Total training steps

        # Initialize data loader with training data
        if self.mid_training:
            # Use TaskMixture for mid-training (SmolTalk, MMLU, GSM8K)
            if self.master_process:
                print("\n" + "=" * 80)
                print("🔄 MID-TRAINING MODE: Using TaskMixture datasets")
                print("=" * 80 + "\n")

            self.train_dataloader = TaskMixtureDataloader(
                data_dir="/sensei-fs/users/divgoyal/nanochat_midtraining_data",
                batch_size=self.config.batch_size,
                block_size=self.config.block_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                split="train",
                master_process=self.master_process,
            )
        else:
            # Use OpenWebText for pretraining
            self.train_dataloader = OpenWebtextDataloader(
                data_dir=f"/sensei-fs/users/divgoyal/openwebtext",
                batch_size=self.config.batch_size,
                block_size=self.config.block_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                split="train",
                master_process=self.master_process,
            )

        # Eval dataloader (only initialize if running evals)
        if self.run_evals:
            self.eval_dataloader = OpenWebtextDataloader(
                data_dir=f"/sensei-fs/users/divgoyal/openwebtext",
                batch_size=self.config.batch_size,
                block_size=self.config.block_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                split="val",
                master_process=self.master_process,
            )
            self.hellaswag_dataloader = HellaSwagDataloader(
                data_dir=f"/sensei-fs/users/divgoyal/hellaswag",
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
                generation_log_file=self.generation_log_file,
                checkpoint_interval=self.config.checkpoint_interval,
                mid_training=self.mid_training,
            )
        else:
            if self.master_process:
                print("Evaluations disabled - skipping eval dataloader initialization")

        # Initialize optimizer with AdamW and weight decay for regularization
        self.optimzer = self.raw_model.configure_optimizers(
            learning_rate=self.max_learning_rate, weight_decay=0.10, device=self.device
        )

        # Initialize wandb for experiment tracking
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

    def _load_checkpoint(self, checkpoint_path):
        """
        Load model weights and optimizer state from a checkpoint.

        Args:
            checkpoint_path (str): Path to the checkpoint file
        """
        if self.master_process:
            print(f"\n{'='*80}")
            print(f"📂 Loading checkpoint from: {checkpoint_path}")
            print(f"{'='*80}\n")

        # weights_only=False required for loading custom classes like GPTConfig
        # (PyTorch 2.6+ defaults to weights_only=True for security)
        checkpoint = torch.load(
            checkpoint_path, map_location=self.device, weights_only=False
        )

        # Load model state
        self.model.load_state_dict(checkpoint["model"])

        # Load config if available
        if "config" in checkpoint:
            self.config = checkpoint["config"]

        # Load step if available (for resuming)
        if "step" in checkpoint:
            self.start_step = checkpoint["step"]
            if self.master_process:
                print(f"✅ Loaded checkpoint from step {self.start_step}")

        # Load val loss if available
        if "val_loss" in checkpoint and self.master_process:
            print(f"   Checkpoint val_loss: {checkpoint['val_loss']:.4f}")

        if self.master_process:
            print(f"{'='*80}\n")

    def get_lr(self, global_step):
        """
        Implement learning rate scheduling with warmup and cosine annealing.

        - Warmup: Linear increase from 0 to max_lr over warmup_steps
        - Cosine annealing: Smooth decay from max_lr to min_lr using cosine function
        - Constant: min_lr after total_steps

        Args:
            global_step (int): Current global training step (across all epochs)

        Returns:
            float: Learning rate for current step
        """
        # Total steps across all epochs
        total_steps = self.max_steps * self.num_epochs

        if global_step < self.warmup_steps:
            # Linear warmup: gradually increase learning rate
            lr = self.max_learning_rate * (global_step + 1) / self.warmup_steps
        elif global_step > total_steps:
            # After total steps, use minimum learning rate
            lr = self.min_learning_rate
        else:
            # Cosine annealing: smooth decay using cosine function
            decay_ratio = (global_step - self.warmup_steps) / (
                total_steps - self.warmup_steps
            )
            assert 0 <= decay_ratio <= 1
            coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))  # Cosine coefficient
            lr = self.min_learning_rate + coeff * (
                self.max_learning_rate - self.min_learning_rate
            )
        return lr

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
            if self.start_step > 0:
                print(f"📍 Resuming from step: {self.start_step:,}")
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
        global_step = (
            self.start_step
        )  # Track global step across all epochs for LR scheduling
        for epoch in range(self.num_epochs):
            # Process all batches in the current epoch
            for step in range(self.start_step, self.max_steps):
                start_time = time.time()  # Track step timing
                self.optimzer.zero_grad()
                loss_accumulator = torch.tensor(0.0, device=self.device)
                for micro_step in range(self.grad_accumulation_steps):
                    # Get training batch and move to device
                    x, y = self.train_dataloader.next_batch()
                    x, y = x.to(self.device), y.to(self.device)

                    # Forward pass: compute predictions and loss
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        logits, loss = self.model(x, y)

                    # normalize loss for gradient accumulation
                    loss = loss / self.grad_accumulation_steps
                    loss_accumulator += loss
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

                # Gradient clipping to prevent exploding gradients
                # This stabilizes training by limiting gradient magnitude
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), 1.0
                )  # Clip gradients to max norm of 1.0

                # Update learning rate based on global step (continuous across epochs)
                lr = self.get_lr(global_step)
                for param_group in self.optimzer.param_groups:
                    param_group["lr"] = lr

                # Apply gradients to update model parameters
                self.optimzer.step()

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
                if self.run_evals and (
                    step % self.run_evals_after == 0 or step == self.max_steps - 1
                ):
                    self.evaluator.estimate_validation_loss(
                        step=step,
                        checkpoint_model=True,
                        max_steps=self.max_steps,
                        epoch=epoch,
                        global_step=global_step,
                    )
                    self.evaluator.estimate_hellaswag_accuracy(
                        step=step, global_step=global_step
                    )
                    self.evaluator.sample_from_model(
                        num_sequences=4,
                        max_length=32,
                        context="Hello, I'm a language model,",
                        step=step,
                    )
                # Log metrics to wandb
                if self.master_process:
                    train_loss = loss_accumulator.item()
                    train_bpb = Evaluators.loss_to_bpb(train_loss)
                    wandb.log(
                        {
                            "epoch": epoch,
                            "epoch_step": step,
                            "step": global_step,
                            "train_loss": train_loss,
                            "train_bpb": train_bpb,
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
                        f"Loss: {train_loss:.4f} | BPB: {train_bpb:.4f} | "
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
