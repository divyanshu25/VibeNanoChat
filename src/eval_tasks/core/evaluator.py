"""
CORE Evaluator for running CORE benchmark tasks during training.
Integrates with the trainer to evaluate model on multiple tasks and log to wandb.
"""

import os
import time

import torch

import wandb
from eval_tasks.core.data_loading import load_core_tasks
from eval_tasks.core.eval_tasks import evaluate_task
from eval_tasks.utils.tokenizer import SimpleTokenizer


class CoreEvaluator:
    """
    Evaluator for CORE benchmark tasks.
    Runs evaluation on multiple tasks and logs results to wandb.
    """

    def __init__(
        self,
        model,
        tokenizer,
        device,
        master_process,
        ddp=False,
        ddp_rank=0,
        ddp_world_size=1,
        eval_bundle_path="resources/eval_bundle",
        tasks_to_run=None,
        max_examples_per_task=None,
    ):
        """
        Initialize CORE evaluator.

        Args:
            model: Model to evaluate
            tokenizer: Tokenizer (tiktoken-based)
            device: Device to run on
            master_process: Whether this is the master process
            ddp: Whether using distributed data parallel
            ddp_rank: Rank of current process
            ddp_world_size: Total number of processes
            eval_bundle_path: Path to eval_bundle directory
            tasks_to_run: Optional list of task labels to run (runs all if None)
            max_examples_per_task: Optional limit on examples per task for faster eval
        """
        self.model = model
        self.device = device
        self.master_process = master_process
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.ddp_world_size = ddp_world_size
        self.max_examples_per_task = max_examples_per_task

        # Wrap tokenizer to match expected interface
        self.tokenizer = SimpleTokenizer(tokenizer, bos_token_id=50256)

        # Load CORE tasks
        yaml_path = os.path.join(eval_bundle_path, "core.yaml")
        if not os.path.exists(yaml_path):
            raise FileNotFoundError(f"CORE config not found: {yaml_path}")

        all_tasks = load_core_tasks(yaml_path, eval_bundle_path)

        # Filter tasks if specified
        if tasks_to_run:
            self.tasks = [t for t in all_tasks if t["label"] in tasks_to_run]
        else:
            self.tasks = all_tasks

        if self.master_process:
            print(f"\n{'='*80}")
            print(f"📋 Loaded {len(self.tasks)} CORE evaluation tasks")
            if self.max_examples_per_task:
                print(
                    f"   Evaluating max {self.max_examples_per_task} examples per task"
                )
            print(f"{'='*80}\n")

    @torch.no_grad()
    def evaluate_all_tasks(self, step=None, global_step=None):
        """
        Evaluate model on all CORE tasks.

        Args:
            step: Current step within epoch
            global_step: Global step across all epochs

        Returns:
            dict: Dictionary of task results
        """
        if not self.master_process:
            # Non-master processes participate in distributed eval but don't log
            pass

        start_time = time.time()
        self.model.eval()

        results = {}

        for i, task in enumerate(self.tasks):
            task_start = time.time()

            # Extract task metadata
            task_meta = {
                "task_type": task["task_type"],
                "num_fewshot": task["num_fewshot"],
                "continuation_delimiter": task["continuation_delimiter"],
            }

            # Evaluate task
            accuracy = evaluate_task(
                model=self.model,
                tokenizer=self.tokenizer,
                data=task["data"],
                device=self.device,
                task_meta=task_meta,
                max_examples=self.max_examples_per_task,
            )

            task_time = time.time() - task_start

            # Store result
            task_label = task["label"]
            if task["num_fewshot"] > 0:
                task_label = f"{task_label}_{task['num_fewshot']}shot"

            results[task_label] = accuracy

            if self.master_process:
                print(
                    f"  [{i+1}/{len(self.tasks)}] {task_label}: {accuracy:.4f} ({task_time:.1f}s)"
                )

        self.model.train()

        elapsed_time = time.time() - start_time

        # Compute average CORE score
        if len(results) > 0:
            core_score = sum(results.values()) / len(results)
        else:
            core_score = 0.0

        if self.master_process:
            print(f"\n{'='*80}")
            print(f"📊 CORE EVALUATION | Step {step if step else 'N/A'}")
            print(f"   Average Score: {core_score:.4f}")
            print(f"   Total Time: {elapsed_time:.2f}s")
            print(f"{'='*80}\n")

            # Log to wandb (only if wandb is initialized)
            try:
                log_dict = {
                    "core_score": core_score,
                }

                # Log individual task results with core/ prefix
                for task_label, accuracy in results.items():
                    log_dict[f"core/{task_label}"] = accuracy

                # Add step information
                if global_step is not None:
                    log_dict["step"] = global_step
                elif step is not None:
                    log_dict["step"] = step

                wandb.log(log_dict)
            except Exception:
                # wandb not initialized (e.g., in test mode), skip logging
                pass

        return results
