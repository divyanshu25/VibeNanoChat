import torch
import torch.distributed as dist
import wandb
import time
import math
from gpt_2.gpt2_model import generate
from gpt_2.utils import accumulate_bpb


class Evaluators:
    def __init__(
        self,
        model,
        eval_dataloader,
        hellaswag_dataloader,
        device,
        master_process,
        ddp,
        ddp_rank=0,
        ddp_world_size=1,
        generation_log_file=None,
        token_bytes_path=None,
    ):
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.hellaswag_dataloader = hellaswag_dataloader
        self.device = device
        self.master_process = master_process
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.generation_log_file = generation_log_file

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
        val_loss_steps = 39  # this number of eval tokens divided by batch size

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

    def estimate_hellaswag_accuracy(self, step, global_step=None):
        """
        Estimate the accuracy of the model on the HellaSwag dataset.
        """
        start_time = time.time()

        self.model.eval()
        self.hellaswag_dataloader.reset()
        hellaswag_accuracy_steps = 79
        hellaswag_accuracy_accumulator = torch.tensor(0.0, device=self.device)
        total_processed_examples = torch.tensor(0.0, device=self.device)
        with torch.no_grad():
            for k in range(hellaswag_accuracy_steps):
                X, Y = self.hellaswag_dataloader.next_batch()
                X = X.to(self.device)
                Y = Y.to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    logits, _ = self.model(X)  # Model returns (logits, loss)

                num_correct = self.hellaswag_dataloader.calculate_correctness(
                    logits, Y, X
                )
                hellaswag_accuracy_accumulator += num_correct
                total_processed_examples += len(Y)

        self.model.train()
        if self.ddp:
            # Sum up correct predictions and total examples across all GPUs
            torch.distributed.all_reduce(
                hellaswag_accuracy_accumulator, op=torch.distributed.ReduceOp.SUM
            )
            torch.distributed.all_reduce(
                total_processed_examples, op=torch.distributed.ReduceOp.SUM
            )
        hellaswag_accuracy = (
            hellaswag_accuracy_accumulator.item() / total_processed_examples.item()
        )

        elapsed_time = time.time() - start_time

        if self.master_process:
            print(f"\n{'='*80}")
            print(
                f"📊 HELLASWAG | Step {step:>5} | Accuracy: {hellaswag_accuracy:.4f} | Time: {elapsed_time:.2f}s"
            )
            print(f"{'='*80}\n")
            wandb.log(
                {
                    "hellaswag_accuracy": hellaswag_accuracy,
                    "step": global_step if global_step is not None else step,
                }
            )

    def sample_from_model(
        self,
        num_sequences=4,
        max_length=32,
        context="Hello, I'm a language model,",
        step=None,
    ):
        if not self.master_process:
            return

        start_time = time.time()

        sample_rng = torch.Generator(device=self.device)
        sample_rng.manual_seed(42 + self.ddp_rank)

        decoded = generate(
            num_sequences=num_sequences,
            max_length=max_length,
            model=self.model,
            context=context,
            device=self.device,
            random_number_generator=sample_rng,
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
                f.write(f"Timestamp: {timestamp} | {step_info}\n")
                f.write(f"Context: {context}\n")
                f.write(f"{'='*80}\n")
                for i, decoded_seq in enumerate(decoded, 1):
                    f.write(f"\nGeneration {i}:\n{decoded_seq}\n")
                f.write(f"\n{'='*80}\n")
