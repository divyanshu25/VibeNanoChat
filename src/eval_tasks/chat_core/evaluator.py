"""
ChatCORE Evaluator for running chat-based generative evaluation tasks during training.

Unlike CORE which uses likelihood-based multiple choice evaluation, ChatCORE
evaluates models by generating completions and checking them for correctness.
This is more realistic for chat models but requires actual text generation.

Supported tasks:
- GSM8K: Math reasoning problems
- HumanEval: Code generation with test execution
- MMLU: Multiple-choice knowledge questions across 57 subjects
- ARC-Easy: Grade-school science questions (easy split)
- ARC-Challenge: Grade-school science questions (hard split, wrong for retrieval models)

-------------------------------------------------------------------------------
Task Examples
-------------------------------------------------------------------------------

GSM8K — multi-step math word problem; evaluated by extracting the number after ####
  Q: Weng earns $12 an hour for babysitting. Yesterday she did 50 minutes.
     How much did she earn?
  A: Weng earns 12/60 = $0.2 per minute.
     Working 50 minutes, she earned 0.2 x 50 = $10.
     #### 10
  If tool use is enabled the model may emit <|python|>12/60<|python_end|> and
  the calculator injects <|output_start|>0.2<|output_end|> before continuing.

HumanEval — function body completion; generated code is executed against test cases
  Prompt:
    def has_close_elements(numbers: List[float], threshold: float) -> bool:
        "" Check if any two numbers in the list are closer than threshold.
        >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
        False
        >>> has_close_elements([1.0, 2.8, 3.0], 0.3)
        True
        ""
  Model fills in the body; pass@1 is computed by running the bundled test suite.

MMLU — 4-choice multiple-choice across 57 subjects; model must output a single letter
  Subject: high_school_physics
  Q: A pendulum has a period of 2 s on Earth. What is its period on the Moon
     where g is 1/6 of Earth's?
  A. 0.8 s   B. 2 s   C. 4.9 s   D. 12 s
  Answer: C

ARC-Easy — grade-school science, solvable by simple retrieval or co-occurrence
  Q: Which of the following is the best way to keep a cold drink cold?
  A. Add ice to it
  B. Keep it in a warm place
  C. Store it in an insulated container
  D. Leave it in direct sunlight
  Answer: A

ARC-Challenge — harder science questions that retrieval-based models answer incorrectly
  Q: Which property of a mineral can be determined just by looking at it?
  A. Its mass   B. Its volume   C. Its luster   D. Its density
  Answer: C

-------------------------------------------------------------------------------
"""

import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist
import wandb

from .kv_cache_utils import (create_kv_cache, forward_pass, get_model_config,
                             prefill_prompt, sample_next_token)
from .tools import use_calculator


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
        use_kv_cache: bool = True,
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
        self.use_kv_cache = use_kv_cache

        # Task registry - will be populated when tasks are loaded
        self.tasks = {}

        # Check if tokenizer supports tool use (python tags)
        self.supports_tools = self._check_tool_support()

        if self.master_process:
            print(f"\n{'='*80}")
            print("💬 ChatCORE Evaluator initialized")
            print(f"   Samples per problem: {num_samples}")
            print(f"   Max tokens: {max_tokens}")
            print(f"   Temperature: {temperature}")
            print(f"   KV cache: {'Enabled' if use_kv_cache else 'Disabled'}")
            print(f"   Tool use enabled: {self.supports_tools}")
            if max_examples:
                print(f"   Max examples per task: {max_examples}")
            print(f"{'='*80}\n")

    def _check_tool_support(self) -> bool:
        """Check if tokenizer has the special tokens needed for tool use."""
        try:
            # Check if tokenizer has _special_tokens attribute
            if not hasattr(self.tokenizer, "_special_tokens"):
                return False

            # Try to get the required special tokens (VibeNanoChat format)
            python_start = self.tokenizer._special_tokens["<|python|>"]
            python_end = self.tokenizer._special_tokens["<|python_end|>"]
            output_start = self.tokenizer._special_tokens["<|output_start|>"]
            output_end = self.tokenizer._special_tokens["<|output_end|>"]

            if self.master_process:
                print(
                    f"python_start: {python_start}, python_end: {python_end}, output_start: {output_start}, output_end: {output_end} found"
                )
            # If we got here, all tokens exist
            return True
        except (AttributeError, KeyError, Exception):
            return False

    def _get_assistant_end_token(self):
        """Get the assistant end token ID."""
        try:
            return self.tokenizer._special_tokens["<|assistant_end|>"]
        except (AttributeError, KeyError, Exception):
            return 50256  # Fallback to GPT-2's <|endoftext|>

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

    def _evaluate_with_details(
        self, eval_fn, example, generated_text, get_details: bool
    ):
        """
        Evaluate a single example and optionally get detailed results.

        Args:
            eval_fn: Evaluation function to call
            example: Example data
            generated_text: Generated completion text
            get_details: Whether to try getting detailed evaluation results

        Returns:
            Tuple of (is_correct: bool, eval_details: Optional[Dict])
        """
        if not get_details:
            is_correct = eval_fn(example, generated_text)
            return is_correct, None

        # Try to get detailed evaluation results
        try:
            eval_result = eval_fn(example, generated_text, return_details=True)
            if isinstance(eval_result, dict):
                is_correct = eval_result.get("success", False)
                eval_details = eval_result
            else:
                is_correct = eval_result
                eval_details = None
        except TypeError:
            # eval_fn doesn't support return_details parameter
            is_correct = eval_fn(example, generated_text)
            eval_details = None

        return is_correct, eval_details

    def _print_execution_result(self, exec_result):
        """Print HumanEval execution result details."""
        print("\n⚙️  EXECUTION RESULT:")
        print(f"   Success: {exec_result.success}")
        if exec_result.timeout:
            print(f"   ⏱️  Timeout: {exec_result.timeout}")
        if exec_result.memory_exceeded:
            print(f"   💾 Memory exceeded: {exec_result.memory_exceeded}")
        if exec_result.error:
            error_msg = exec_result.error[:200]
            if len(exec_result.error) > 200:
                error_msg += "..."
            print(f"   ❗ Error: {error_msg}")
        if exec_result.stdout and exec_result.stdout.strip():
            stdout_msg = exec_result.stdout[:100]
            if len(exec_result.stdout) > 100:
                stdout_msg += "..."
            print(f"   📤 Stdout: {stdout_msg}")
        if exec_result.stderr and exec_result.stderr.strip():
            stderr_msg = exec_result.stderr[:100]
            if len(exec_result.stderr) > 100:
                stderr_msg += "..."
            print(f"   ⚠️  Stderr: {stderr_msg}")

    def _print_answer_comparison(self, eval_details):
        """Print GSM8K answer comparison details."""
        print("\n🔢 ANSWER COMPARISON:")
        print(f"   Expected: {eval_details['reference_answer']}")
        print(f"   Predicted: {eval_details['predicted_answer']}")
        match = eval_details["reference_answer"] == eval_details["predicted_answer"]
        print(f"   Match: {match}")

    def _print_detailed_example(
        self,
        example,
        generated_text,
        is_correct,
        eval_details,
        total_count,
        num_examples,
        prompt_tokens=None,
    ):
        """Print detailed output for a single evaluation example."""
        print(f"\n{'='*80}")
        print(f"📝 Example {total_count}/{num_examples//self.ddp_world_size}")
        print(f"{'='*80}")

        # Show decoded prompt tokens if available
        if prompt_tokens is not None:
            print("🔵 PROMPT (decoded from tokens):")
            print(f"{'='*80}")
            decoded_prompt = self.tokenizer.decode(prompt_tokens)
            print(f"{decoded_prompt}")
            print(f"{'='*80}")
        # Fallback to showing prompt or question from example
        elif "prompt" in example:
            print("🔵 PROMPT:")
            print(f"{'='*80}")
            print(f"{example['prompt']}")
            print(f"{'='*80}")
        elif "question" in example:
            print("🔵 QUESTION:")
            print(f"{'='*80}")
            print(f"{example['question']}")
            print(f"{'='*80}")

        # Show generated output
        print("\n🤖 MODEL GENERATED:")
        print(f"{'='*80}")
        print(f"{generated_text}")
        print(f"{'='*80}")
        # Show evaluation details if available
        if eval_details:
            # HumanEval execution details
            if "result" in eval_details:
                self._print_execution_result(eval_details["result"])

            # GSM8K answer comparison
            if (
                "reference_answer" in eval_details
                and "predicted_answer" in eval_details
            ):
                self._print_answer_comparison(eval_details)

        # Show result
        print(f"\n{'✅ CORRECT' if is_correct else '❌ INCORRECT'}")
        print(f"{'='*80}\n")

    @torch.no_grad()
    def generate_completion(self, prompt_tokens: List[int]) -> str:
        """
        Generate a single completion from prompt tokens using KV caching.

        KV Caching Optimization:
        Instead of reprocessing all tokens at each step, we use a two-phase approach:
        1. PREFILL PHASE: Process all prompt tokens at once, cache their K/V
        2. DECODE PHASE: Generate one token at a time, reusing cached K/V

        This reduces computation from O(N²) to O(N), making generation 5-10x faster.

        Args:
            prompt_tokens: List of token IDs for the prompt

        Returns:
            Generated text (decoded tokens)
        """
        # Setup: Get model config and special tokens
        num_heads, head_dim, num_layers, max_seq_len = get_model_config(self.model)
        assistant_end_id = self._get_assistant_end_token()

        # Create KV cache for the entire generation (if enabled)
        kv_cache = create_kv_cache(
            len(prompt_tokens),
            self.max_tokens,
            num_heads,
            head_dim,
            num_layers,
            max_seq_len,
            self.use_kv_cache,
        )

        # PHASE 1: PREFILL - Process entire prompt at once
        next_token_logits = prefill_prompt(
            self.model, prompt_tokens, kv_cache, self.device
        )

        generated_tokens = list(prompt_tokens)

        for _ in range(self.max_tokens):
            next_token = sample_next_token(
                next_token_logits, self.temperature, self.top_k
            )
            if next_token == assistant_end_id:
                break

            generated_tokens.append(next_token)
            if (
                hasattr(self.model, "max_seq_len")
                and len(generated_tokens) >= self.model.max_seq_len
            ):
                break

            next_token_logits = forward_pass(
                self.model,
                next_token if self.use_kv_cache else generated_tokens,
                kv_cache,
                self.device,
            )

        new_tokens = generated_tokens[len(prompt_tokens) :]
        return self.tokenizer.decode(new_tokens)

    @torch.no_grad()
    def generate_completion_with_tools(self, prompt_tokens: List[int]) -> str:
        """
        Generate a completion with calculator tool use support using KV caching.

        When the model emits <|python|>expr<|python_end|>, the expression is
        evaluated with use_calculator() and the result is immediately injected
        as <|output_start|>result<|output_end|> before sampling resumes.

        Example:
            Model: "She earns <|python|>12/60<|python_end|>"
            → Calculator: 12/60 = 0.2
            → Injected: "<|output_start|>0.2<|output_end|>"
            → Model continues: " per minute..."

        Args:
            prompt_tokens: List of token IDs for the prompt

        Returns:
            Generated text (decoded tokens)
        """
        if not self.supports_tools:
            return self.generate_completion(prompt_tokens)

        python_start = self.tokenizer._special_tokens["<|python|>"]
        python_end = self.tokenizer._special_tokens["<|python_end|>"]
        output_start = self.tokenizer._special_tokens["<|output_start|>"]
        output_end = self.tokenizer._special_tokens["<|output_end|>"]
        assistant_end_id = self._get_assistant_end_token()

        num_heads, head_dim, num_layers, max_seq_len = get_model_config(self.model)
        kv_cache = create_kv_cache(
            len(prompt_tokens),
            self.max_tokens,
            num_heads,
            head_dim,
            num_layers,
            max_seq_len,
            self.use_kv_cache,
        )

        next_token_logits = prefill_prompt(
            self.model, prompt_tokens, kv_cache, self.device
        )

        generated_tokens = list(prompt_tokens)
        python_expr_tokens: List[int] = []
        in_python_block = False

        def _feed(token: int) -> Optional[torch.Tensor]:
            """Append token, run one decode step; returns None if seq limit reached."""
            generated_tokens.append(token)
            if (
                hasattr(self.model, "max_seq_len")
                and len(generated_tokens) >= self.model.max_seq_len
            ):
                return None
            if self.use_kv_cache:
                return forward_pass(self.model, token, kv_cache, self.device)
            return forward_pass(self.model, generated_tokens, kv_cache, self.device)

        for _ in range(self.max_tokens):
            next_token = sample_next_token(
                next_token_logits, self.temperature, self.top_k
            )
            if next_token == assistant_end_id:
                break

            # State machine: update mode and compute any tokens to inject after this one
            injection: List[int] = []
            if next_token == python_start:
                in_python_block = True
                python_expr_tokens = []
            elif next_token == python_end and in_python_block:
                in_python_block = False
                if python_expr_tokens:
                    result = use_calculator(self.tokenizer.decode(python_expr_tokens))
                    if result is not None:
                        injection = (
                            [output_start]
                            + self.tokenizer.encode(str(result))
                            + [output_end]
                        )
                python_expr_tokens = []
            elif in_python_block:
                python_expr_tokens.append(next_token)

            # Feed the sampled token through the model
            next_token_logits = _feed(next_token)
            if next_token_logits is None:
                break

            for tok in injection:
                next_token_logits = _feed(tok)
                if next_token_logits is None:
                    break
            else:
                continue
            break  # seq limit hit inside injection loop

        new_tokens = generated_tokens[len(prompt_tokens) :]
        return self.tokenizer.decode(new_tokens)

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
        show_examples = 5  # Show first 5 examples with detailed output

        # Distribute examples across ranks
        for idx in range(self.ddp_rank, num_examples, self.ddp_world_size):
            example = data[idx]

            # Render prompt from conversation
            prompt_tokens = render_fn(example)

            # Generate completion (with tool use if supported)
            try:
                if self.supports_tools:
                    generated_text = self.generate_completion_with_tools(prompt_tokens)
                else:
                    generated_text = self.generate_completion(prompt_tokens)

                # Evaluate correctness (with details for first few examples)
                get_details = self.master_process and total_count < show_examples
                is_correct, eval_details = self._evaluate_with_details(
                    eval_fn, example, generated_text, get_details
                )

                if is_correct:
                    correct_count += 1
                total_count += 1

                # Show detailed output for first few examples (only on master)
                if self.master_process and total_count <= show_examples:
                    self._print_detailed_example(
                        example,
                        generated_text,
                        is_correct,
                        eval_details,
                        total_count,
                        num_examples,
                        prompt_tokens,
                    )

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
            "num_evaluated": total_count,
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
                    # Log individual task results with chatcore/ prefix
                    try:
                        log_dict = {
                            f"chatcore/{task_name}": results["accuracy"],
                            "step": global_step if global_step is not None else step,
                        }
                        wandb.log(log_dict)
                    except Exception:
                        # wandb not initialized, skip logging
                        pass

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
                    "step": global_step if global_step is not None else step,
                }
                wandb.log(log_dict)
            except Exception:
                # wandb not initialized, skip logging
                pass

        return all_results
