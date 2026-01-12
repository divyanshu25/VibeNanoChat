"""
ChatCORE Evaluator for running chat-based generative evaluation tasks during training.

Unlike CORE which uses likelihood-based multiple choice evaluation, ChatCORE
evaluates models by generating completions and checking them for correctness.
This is more realistic for chat models but requires actual text generation.

Supported tasks:
- GSM8K: Math reasoning problems
- (Future: MMLU, ARC, HumanEval, SpellingBee)
"""

import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

import wandb


class ChatCoreEvaluator:
    """
    Evaluator for ChatCORE benchmark tasks (generative chat evaluation).

    This evaluator:
    1. Loads chat-based evaluation tasks (GSM8K, etc.)
    2. Generates completions for each problem
    3. Evaluates completions for correctness
    4. Computes pass@k metrics
    5. Logs results to wandb
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        master_process: bool,
        ddp: bool = False,
        ddp_rank: int = 0,
        ddp_world_size: int = 1,
        max_examples: Optional[int] = None,
        num_samples: int = 1,
        max_tokens: int = 512,
        temperature: float = 0.0,
        top_k: int = 50,
    ):
        """
        Initialize ChatCORE evaluator.

        Args:
            model: Model to evaluate (must have a generate or similar method)
            tokenizer: Tokenizer with encode/decode methods
            device: Device to run on
            master_process: Whether this is the master process for logging
            ddp: Whether using distributed data parallel
            ddp_rank: Rank of current process
            ddp_world_size: Total number of processes
            max_examples: Optional limit on examples per task
            num_samples: Number of samples to generate per problem (for pass@k)
            max_tokens: Maximum tokens to generate per completion
            temperature: Sampling temperature (0.0 = greedy)
            top_k: Top-k sampling parameter
        """
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.master_process = master_process
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.max_examples = max_examples
        self.num_samples = num_samples
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_k = top_k

        # Task registry - will be populated when tasks are loaded
        self.tasks = {}

        if self.master_process:
            print(f"\n{'='*80}")
            print("💬 ChatCORE Evaluator initialized")
            print(f"   Samples per problem: {num_samples}")
            print(f"   Max tokens: {max_tokens}")
            print(f"   Temperature: {temperature}")
            if max_examples:
                print(f"   Max examples per task: {max_examples}")
            print(f"{'='*80}\n")

    def register_task(self, task_name: str, task_config: Dict):
        """
        Register a task for evaluation.

        Args:
            task_name: Name of the task (e.g., "GSM8K")
            task_config: Task configuration dict with keys:
                - 'load_fn': Function to load data
                - 'eval_fn': Function to evaluate predictions
                - 'render_fn': Function to render prompt from conversation
        """
        self.tasks[task_name] = task_config

        if self.master_process:
            print(f"✓ Registered task: {task_name}")

    @torch.no_grad()
    def generate_completion(self, prompt_tokens: List[int]) -> str:
        """
        Generate a single completion from prompt tokens.

        Args:
            prompt_tokens: List of token IDs for the prompt

        Returns:
            Generated text (decoded tokens)
        """
        # Convert prompt to tensor
        prompt_tensor = torch.tensor(
            [prompt_tokens], dtype=torch.long, device=self.device
        )

        # Generate tokens
        # This is a simple greedy/sampling generation loop
        # You may need to adapt this to your model's specific generation API
        generated_tokens = list(prompt_tokens)

        for _ in range(self.max_tokens):
            # Get logits for next token
            with torch.amp.autocast(
                device_type=(
                    self.device.type if hasattr(self.device, "type") else "cuda"
                ),
                dtype=torch.bfloat16,
            ):
                logits = self.model(prompt_tensor)

            # Handle tuple output (logits, loss)
            if isinstance(logits, tuple):
                logits = logits[0]

            # Get logits for last position
            next_token_logits = logits[0, -1, :]

            # Apply temperature
            if self.temperature > 0:
                next_token_logits = next_token_logits / self.temperature

                # Apply top-k filtering
                if self.top_k > 0:
                    top_k_logits, top_k_indices = torch.topk(
                        next_token_logits, self.top_k
                    )
                    next_token_logits = torch.full_like(
                        next_token_logits, float("-inf")
                    )
                    next_token_logits.scatter_(0, top_k_indices, top_k_logits)

                # Sample
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                # Greedy
                next_token = next_token_logits.argmax().item()

            # Check for end of sequence
            # Common EOS token IDs: 50256 (GPT-2), 2 (Llama), etc.
            # You may need to adapt this based on your tokenizer
            if hasattr(self.tokenizer, "eos_token_id"):
                eos_id = self.tokenizer.eos_token_id
            else:
                eos_id = 50256  # GPT-2 default

            if next_token == eos_id:
                break

            generated_tokens.append(next_token)

            # Update prompt tensor for next iteration
            prompt_tensor = torch.tensor(
                [generated_tokens], dtype=torch.long, device=self.device
            )

            # Truncate if exceeds model's max sequence length
            if (
                hasattr(self.model, "max_seq_len")
                and len(generated_tokens) >= self.model.max_seq_len
            ):
                break

        # Decode only the newly generated tokens (exclude prompt)
        new_tokens = generated_tokens[len(prompt_tokens) :]
        generated_text = self.tokenizer.decode(new_tokens)

        return generated_text

    @torch.no_grad()
    def evaluate_task(self, task_name: str) -> Dict[str, float]:
        """
        Evaluate model on a single task.

        Args:
            task_name: Name of the task to evaluate

        Returns:
            Dict with evaluation metrics (e.g., 'accuracy', 'pass@1', etc.)
        """
        if task_name not in self.tasks:
            raise ValueError(f"Task {task_name} not registered")

        task_config = self.tasks[task_name]

        # Load data
        data = task_config["load_fn"](max_examples=self.max_examples)
        eval_fn = task_config["eval_fn"]
        render_fn = task_config["render_fn"]

        num_examples = len(data)

        if self.master_process:
            print(f"\n{'='*60}")
            print(f"Evaluating {task_name}: {num_examples} examples")
            print(f"{'='*60}")

        # Track results across all ranks
        correct_count = 0
        total_count = 0

        # Distribute examples across ranks
        for idx in range(self.ddp_rank, num_examples, self.ddp_world_size):
            example = data[idx]

            # Render prompt from conversation
            prompt_tokens = render_fn(example)

            # Generate completion
            try:
                generated_text = self.generate_completion(prompt_tokens)

                # Evaluate correctness
                is_correct = eval_fn(example, generated_text)

                if is_correct:
                    correct_count += 1
                total_count += 1

                # Progress logging (only on master)
                if self.master_process and total_count % 10 == 0:
                    acc = correct_count / total_count if total_count > 0 else 0
                    print(
                        f"  Progress: {total_count}/{num_examples//self.ddp_world_size} | Accuracy: {acc:.3f}"
                    )

            except Exception as e:
                if self.master_process:
                    print(f"  Warning: Failed to evaluate example {idx}: {e}")
                continue

        # Aggregate results across all ranks
        if self.ddp_world_size > 1:
            correct_tensor = torch.tensor(
                [correct_count], dtype=torch.long, device=self.device
            )
            total_tensor = torch.tensor(
                [total_count], dtype=torch.long, device=self.device
            )

            dist.barrier()
            dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
            dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

            correct_count = correct_tensor.item()
            total_count = total_tensor.item()

        # Compute metrics
        accuracy = correct_count / total_count if total_count > 0 else 0.0

        if self.master_process:
            print(
                f"\n  ✓ {task_name} Accuracy: {accuracy:.4f} ({correct_count}/{total_count})"
            )

        return {
            "accuracy": accuracy,
            "correct": correct_count,
            "total": total_count,
        }

    @torch.no_grad()
    def evaluate_all_tasks(
        self, step: Optional[int] = None, global_step: Optional[int] = None
    ) -> Dict[str, float]:
        """
        Evaluate model on all registered tasks.

        Args:
            step: Current step within epoch
            global_step: Global step across all epochs

        Returns:
            Dict of results for all tasks
        """
        start_time = time.time()
        self.model.eval()

        all_results = {}

        for task_name in self.tasks.keys():
            task_start = time.time()

            try:
                results = self.evaluate_task(task_name)
                all_results[task_name] = results

                task_time = time.time() - task_start

                if self.master_process:
                    print(
                        f"  {task_name}: {results['accuracy']:.4f} ({task_time:.1f}s)"
                    )

            except Exception as e:
                if self.master_process:
                    print(f"  ✗ Failed to evaluate {task_name}: {e}")
                continue

        self.model.train()

        elapsed_time = time.time() - start_time

        # Compute average ChatCORE score
        if len(all_results) > 0:
            chatcore_score = sum(r["accuracy"] for r in all_results.values()) / len(
                all_results
            )
        else:
            chatcore_score = 0.0

        if self.master_process:
            print(f"\n{'='*80}")
            print(f"💬 ChatCORE EVALUATION | Step {step if step else 'N/A'}")
            print(f"   Average Score: {chatcore_score:.4f}")
            print(f"   Total Time: {elapsed_time:.2f}s")
            print(f"{'='*80}\n")

            # Log to wandb
            try:
                log_dict = {
                    "chatcore_score": chatcore_score,
                }

                # Log individual task results with chatcore/ prefix
                for task_name, results in all_results.items():
                    log_dict[f"chatcore/{task_name}"] = results["accuracy"]

                # Add step information
                if global_step is not None:
                    log_dict["step"] = global_step
                elif step is not None:
                    log_dict["step"] = step

                wandb.log(log_dict)
            except Exception:
                # wandb not initialized, skip logging
                pass

        return all_results
