"""
CORE Evaluator for running CORE benchmark tasks during training.
Integrates with the trainer to evaluate model on multiple tasks and log to wandb.
"""

import csv
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
        self.tokenizer = SimpleTokenizer(tokenizer, bos_token_id=50257)

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

        # Load random baselines for centering scores
        self.random_baselines = self._load_random_baselines(eval_bundle_path)

        if self.master_process:
            print(f"\n{'='*80}")
            print(f"📋 Loaded {len(self.tasks)} CORE evaluation tasks")
            if self.max_examples_per_task:
                print(
                    f"   Evaluating max {self.max_examples_per_task} examples per task"
                )
            print(f"{'='*80}\n")

    def _load_random_baselines(self, eval_bundle_path):
        """
        Load random baseline values from eval_meta_data.csv for centering scores.

        Returns:
            dict: Mapping from task label to random baseline percentage
        """
        csv_path = os.path.join(eval_bundle_path, "eval_meta_data.csv")
        if not os.path.exists(csv_path):
            if self.master_process:
                print(f"⚠️  Warning: eval_meta_data.csv not found at {csv_path}")
                print("   Will use raw scores without centering")
            return {}

        random_baselines = {}
        with open(csv_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                task_name = row["Eval Task"]
                random_baseline = row["Random baseline"]
                random_baselines[task_name] = float(random_baseline)

        return random_baselines

    @torch.no_grad()
    def evaluate_all_tasks(self, step=None, global_step=None, flops_so_far=None):
        """
        Evaluate model on all CORE tasks.

        Args:
            step: Current step within epoch
            global_step: Global step across all epochs
            flops_so_far: Total FLOPs used in training

        Returns:
            dict: Dictionary of task results (raw accuracies)
        """
        if not self.master_process:
            # Non-master processes participate in distributed eval but don't log
            pass

        start_time = time.time()
        self.model.eval()

        results = {}
        centered_results = {}

        for i, task in enumerate(self.tasks):
            task_start = time.time()

            # Extract task metadata
            task_meta = {
                "task_type": task["task_type"],
                "num_fewshot": task["num_fewshot"],
                "continuation_delimiter": task["continuation_delimiter"],
            }

            # Evaluate task
            accuracy, num_evaluated = evaluate_task(
                model=self.model,
                tokenizer=self.tokenizer,
                data=task["data"],
                device=self.device,
                task_meta=task_meta,
                max_examples=self.max_examples_per_task,
            )

            task_time = time.time() - task_start

            # Store raw result
            task_label = task["label"]
            if task["num_fewshot"] > 0:
                task_label = f"{task_label}_{task['num_fewshot']}shot"

            results[task_label] = accuracy

            # Apply centering if random baseline is available
            # Centering formula: (accuracy - random_baseline) / (1.0 - random_baseline)
            # This adjusts the score to account for random guessing
            base_label = task["label"]
            if base_label in self.random_baselines:
                random_baseline = self.random_baselines[base_label]
                # Convert percentage to decimal (e.g., 25 -> 0.25)
                random_baseline_decimal = 0.01 * random_baseline
                centered_result = (accuracy - random_baseline_decimal) / (
                    1.0 - random_baseline_decimal
                )
                centered_results[task_label] = centered_result
            else:
                # No random baseline available, use raw accuracy as centered result
                centered_results[task_label] = accuracy

            if self.master_process:
                centered_str = (
                    f"centered: {centered_results[task_label]:.4f}"
                    if task_label in centered_results
                    else ""
                )
                print(
                    f"  [{i+1}/{len(self.tasks)}] {task_label}: {accuracy:.4f} {centered_str} ({num_evaluated} examples, {task_time:.1f}s)"
                )

        self.model.train()

        elapsed_time = time.time() - start_time

        # Compute average CORE score from CENTERED results (matching nanochat methodology)
        if len(centered_results) > 0:
            core_score = sum(centered_results.values()) / len(centered_results)
        else:
            core_score = 0.0

        if self.master_process:
            print(f"\n{'='*80}")
            print(f"📊 CORE EVALUATION | Step {step if step else 'N/A'}")
            print(f"   Core Score (centered): {core_score:.4f}")
            print(f"   Total Time: {elapsed_time:.2f}s")
            print(f"{'='*80}\n")

            # Log to wandb (only if wandb is initialized)
            try:
                log_dict = {
                    "core_score": core_score,
                    "total_training_flops_core": flops_so_far,
                }

                # Log individual task results (both raw and centered)
                for task_label, accuracy in results.items():
                    log_dict[f"core/{task_label}"] = accuracy
                    if task_label in centered_results:
                        log_dict[f"core/{task_label}_centered"] = centered_results[
                            task_label
                        ]

                # Add step information
                if global_step is not None:
                    log_dict["step"] = global_step
                elif step is not None:
                    log_dict["step"] = step

                wandb.log(log_dict)
            except Exception:
                # wandb not initialized (e.g., in test mode), skip logging
                pass

        # Return both raw results and centered core score for logging purposes
        # The core_score now matches nanochat's methodology (averaged centered results)
        return results, core_score, centered_results
