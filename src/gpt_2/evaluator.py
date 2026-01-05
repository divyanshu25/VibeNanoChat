import torch
import wandb
import time
import math
from gpt_2.gpt2_model import generate


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
        generation_log_file=None,
    ):
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.hellaswag_dataloader = hellaswag_dataloader
        self.device = device
        self.master_process = master_process
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.generation_log_file = generation_log_file

    @staticmethod
    def loss_to_bpb(loss, bytes_per_token=4.0):
        """
        Convert cross-entropy loss (in nats) to bits per byte (BPB).

        Args:
            loss: Cross-entropy loss value
            bytes_per_token: Average bytes per token (default 4.0 for GPT-2 on English)

        Returns:
            float: Bits per byte metric

        Formula: BPB = loss × log₂(e) / bytes_per_token
        Lower BPB is better (perfect prediction = 0, random = ~8)
        """
        bits_per_token = loss * math.log2(math.e)
        return bits_per_token / bytes_per_token

    def estimate_validation_loss(self, step, global_step=None):
        """
        Estimate average loss on validation set.
        This provides a more stable estimate than single-batch loss.

        Args:
            step: Current step within epoch
            global_step: Global step across all epochs (for wandb logging)

        Returns:
            float: Validation loss
        """
        start_time = time.time()

        self.model.eval()
        val_loss_accumulator = torch.tensor(0.0, device=self.device)
        self.eval_dataloader.reset()
        val_loss_steps = 39  # this number of eval tokens divided by batch size

        with torch.no_grad():
            for k in range(val_loss_steps):
                X, Y = self.eval_dataloader.next_batch()
                X = X.to(self.device)
                Y = Y.to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    _, loss = self.model(X, Y)
                loss = loss / val_loss_steps  # shape: (1, 1)
                val_loss_accumulator += loss

        self.model.train()
        if self.ddp:
            torch.distributed.all_reduce(
                val_loss_accumulator, op=torch.distributed.ReduceOp.AVG
            )

        elapsed_time = time.time() - start_time

        # Calculate BPB (bits per byte) from loss
        val_loss = val_loss_accumulator.item()
        val_bpb = self.loss_to_bpb(val_loss)

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
