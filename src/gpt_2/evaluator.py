import torch
import wandb
import os
from gpt_2.gpt2_model import generate


class Evaluators:
    def __init__(self, model, eval_dataloader, device, master_process, ddp, ddp_rank=0):
        self.model = model
        self.eval_dataloader = eval_dataloader
        self.device = device
        self.master_process = master_process
        self.ddp = ddp
        self.ddp_rank = ddp_rank

    def estimate_validation_loss(self, step, checkpoint_model=False, max_steps=None):
        """
        Estimate average loss on both training and validation sets.
        This provides a more stable estimate than single-batch loss.

        Returns:
            dict: Contains 'train' and 'val' average losses
        """
        self.model.eval()
        val_loss_accumulator = torch.tensor(0.0, device=self.device)
        self.eval_dataloader.reset()
        val_loss_steps = 34

        with torch.no_grad():
            for k in range(val_loss_steps):
                X, Y = self.eval_dataloader.next_batch()
                X = X.to(self.device)
                Y = Y.to(self.device)
                with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                    _, loss = self.model(X, Y)
                loss = loss / val_loss_steps
                val_loss_accumulator += loss

        self.model.train()

        if self.ddp:
            torch.distributed.all_reduce(
                val_loss_accumulator, op=torch.distributed.ReduceOp.AVG
            )

        if self.master_process:
            print(f"\n{'='*80}")
            print(
                f"📊 VALIDATION | Step {step:>5} | Val Loss: {val_loss_accumulator.item():.4f}"
            )
            print(f"{'='*80}\n")
            wandb.log({"val_loss": val_loss_accumulator.item(), "step": step})

        if self.master_process and (
            (checkpoint_model and step > 0 and step % 5000 == 0)
            or step == max_steps - 1
        ):
            checkpoint = {
                "model": self.model.state_dict(),
                "config": self.model.config,
                "step": step,
                "val_loss": val_loss_accumulator.item(),
            }
            checkpoint_dir = "/sensei-fs/users/divgoyal/nanogpt/checkpoints"
            os.makedirs(checkpoint_dir, exist_ok=True)
            checkpoint_path = f"{checkpoint_dir}/model_checkpoint_{step}.pt"
            torch.save(checkpoint, checkpoint_path)
            print(f"\n💾 Checkpoint saved: {checkpoint_path}\n")

    def sample_from_model(
        self, num_sequences=4, max_length=32, context="Hello, I'm a language model,"
    ):
        if not self.master_process:
            return

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

        print(f"🎯 SAMPLE GENERATIONS:")
        for i, decoded_seq in enumerate(decoded, 1):
            # Truncate if too long and add ellipsis
            display_text = (
                decoded_seq if len(decoded_seq) <= 100 else decoded_seq[:100] + "..."
            )
            print(f"  {i}. {display_text}")
        print()
