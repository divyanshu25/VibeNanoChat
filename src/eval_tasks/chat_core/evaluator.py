"""
ChatCORE Evaluator for running chat-based generative evaluation tasks during training.

Unlike CORE which uses likelihood-based multiple choice evaluation, ChatCORE
evaluates models by generating completions and checking them for correctness.
This is more realistic for chat models but requires actual text generation.

Supported tasks:
- GSM8K: Math reasoning problems
- (Future: MMLU, ARC, HumanEval, SpellingBee)

"""

import os
# Import KVCache from the model package (where it belongs as a core component)
import sys
import time
from typing import Dict, List, Optional

import torch
import torch.distributed as dist

import wandb

from .tools import use_calculator

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "../../"))
from gpt_2.kv_cache import KVCache


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

            # Try to get the required special tokens (NanoGPT format)
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
        # =====================================================================
        # SETUP: Extract model configuration for KV cache dimensions
        # =====================================================================
        # Get model dimensions needed for KV cache
        if hasattr(self.model, "config"):
            m = self.model.config
            num_heads = m.n_kv_head if hasattr(m, "n_kv_head") else m.n_head
            head_dim = m.n_embed // m.n_head
            num_layers = m.n_layer
            max_seq_len = m.block_size if hasattr(m, "block_size") else 2048
        else:
            # Fallback: try to infer from model structure
            print("Warning: Could not find model.config, using fallback dimensions")
            num_heads = 12
            head_dim = 64
            num_layers = 12
            max_seq_len = 2048

        # Get termination token
        try:
            assistant_end_id = self.tokenizer._special_tokens["<|assistant_end|>"]
        except (AttributeError, KeyError, Exception):
            assistant_end_id = 50256  # Fallback to GPT-2's <|endoftext|>

        # =====================================================================
        # SETUP: Create KV cache for the entire generation (if enabled)
        # =====================================================================
        # Create ONE cache that's large enough for prompt + generation
        # This avoids the overhead of creating two caches and copying between them

        kv_cache = None
        if self.use_kv_cache:
            estimated_length = len(prompt_tokens) + self.max_tokens
            if estimated_length > max_seq_len:
                estimated_length = max_seq_len

            kv_cache = KVCache(
                batch_size=1,
                num_heads=num_heads,
                seq_len=estimated_length,
                head_dim=head_dim,
                num_layers=num_layers,
            )

        # =====================================================================
        # PHASE 1: PREFILL - Process entire prompt at once
        # =====================================================================
        # This processes all prompt tokens in a single forward pass and caches
        # their Key and Value tensors. This is much more efficient than processing
        # them one-by-one.

        # Process all prompt tokens at once
        prompt_tensor = torch.tensor(
            [prompt_tokens], dtype=torch.long, device=self.device
        )

        with torch.amp.autocast(
            device_type=(
                self.device.type if hasattr(self.device, "type") else str(self.device)
            ),
            dtype=torch.bfloat16,
        ):
            # Forward pass with KV cache - this caches K,V for all prompt tokens
            logits, _ = self.model(prompt_tensor, kv_cache=kv_cache)

        # Get logits for the last position (what comes after the prompt)
        next_token_logits = logits[0, -1, :]

        # =====================================================================
        # PHASE 2: DECODE - Generate tokens one at a time
        # =====================================================================
        # Now we generate tokens autoregressively, passing only ONE new token
        # at a time. The KV cache already contains all previous tokens' K,V.

        # Track generated tokens (starts with prompt)
        generated_tokens = list(prompt_tokens)

        # =====================================================================
        # MAIN GENERATION LOOP
        # =====================================================================
        for _ in range(self.max_tokens):
            # -----------------------------------------------------------------
            # STEP 1: Sample next token from logits
            # -----------------------------------------------------------------
            if self.temperature > 0:
                # Apply temperature scaling
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

                # Sample from probability distribution
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                # Greedy decoding (deterministic)
                next_token = next_token_logits.argmax().item()

            # -----------------------------------------------------------------
            # STEP 2: Check for termination
            # -----------------------------------------------------------------
            if next_token == assistant_end_id:
                break

            # Add token to sequence
            generated_tokens.append(next_token)

            # Check sequence length limit
            if (
                hasattr(self.model, "max_seq_len")
                and len(generated_tokens) >= self.model.max_seq_len
            ):
                break

            # -----------------------------------------------------------------
            # STEP 3: Get logits for next token (KEY OPTIMIZATION)
            # -----------------------------------------------------------------
            # With KV cache: we only pass the ONE new token, not the entire
            # sequence! The KV cache contains all previous context.
            # This makes each step O(1) instead of O(N).
            #
            # Without KV cache: we pass the entire sequence each time (slower)

            if self.use_kv_cache:
                # KV cache enabled: pass only new token
                next_token_tensor = torch.tensor(
                    [[next_token]], dtype=torch.long, device=self.device
                )
            else:
                # No KV cache: pass entire sequence
                next_token_tensor = torch.tensor(
                    [generated_tokens], dtype=torch.long, device=self.device
                )

            with torch.amp.autocast(
                device_type=(
                    self.device.type
                    if hasattr(self.device, "type")
                    else str(self.device)
                ),
                dtype=torch.bfloat16,
            ):
                # Forward pass with or without KV cache
                logits, _ = self.model(next_token_tensor, kv_cache=kv_cache)

            # Extract logits for next prediction
            next_token_logits = logits[0, -1, :]

        # =====================================================================
        # FINALIZATION: Decode and return generated text
        # =====================================================================
        # Extract only newly generated tokens (exclude the prompt)
        new_tokens = generated_tokens[len(prompt_tokens) :]
        generated_text = self.tokenizer.decode(new_tokens)

        return generated_text

    @torch.no_grad()
    def generate_completion_with_tools(self, prompt_tokens: List[int]) -> str:
        """
        Generate a completion with calculator tool use support using KV caching.

        This implements a tool-use state machine that allows the model to execute
        Python expressions during generation for accurate mathematical computation.

        KV Caching with Tool Use:
        We use the same two-phase approach (prefill + decode), but with added
        complexity from forced tokens (calculator results). When forcing tokens,
        we still update the KV cache so the model sees them in context.

        Flow:
        1. PREFILL: Process prompt tokens at once, cache K/V
        2. DECODE: Generate tokens one at a time with KV cache
        3. When model emits <|python|>, enter "python mode"
        4. Collect tokens until <|python_end|> as a Python expression
        5. Execute the expression with use_calculator()
        6. Force result: <|output_start|>result<|output_end|>
        7. Continue generation with forced tokens in context

        Example generation sequence:
            Model: "She earns <|python|>12/60<|python_end|>"
            → Calculator executes: 12/60 = 0.2
            → Forced output: "<|output_start|>0.2<|output_end|>"
            → Model continues: " per minute..."

        Args:
            prompt_tokens: List of token IDs for the prompt

        Returns:
            Generated text (decoded tokens)
        """
        # =====================================================================
        # SETUP: Check tool support and get special tokens
        # =====================================================================
        if not self.supports_tools:
            # Fall back to regular generation if tokenizer lacks tool tokens
            return self.generate_completion(prompt_tokens)

        # Get the special tokens that coordinate the tool-use state machine
        python_start = self.tokenizer._special_tokens["<|python|>"]
        python_end = self.tokenizer._special_tokens["<|python_end|>"]
        output_start = self.tokenizer._special_tokens["<|output_start|>"]
        output_end = self.tokenizer._special_tokens["<|output_end|>"]

        # Get termination token: <|assistant_end|>
        try:
            assistant_end_id = self.tokenizer._special_tokens["<|assistant_end|>"]
        except (AttributeError, KeyError, Exception):
            assistant_end_id = 50256  # Fallback to GPT-2's <|endoftext|>

        # =====================================================================
        # SETUP: Extract model configuration for KV cache dimensions
        # =====================================================================
        if hasattr(self.model, "config"):
            m = self.model.config
            num_heads = m.n_kv_head if hasattr(m, "n_kv_head") else m.n_head
            head_dim = m.n_embed // m.n_head
            num_layers = m.n_layer
            max_seq_len = m.block_size if hasattr(m, "block_size") else 2048
        else:
            # Fallback dimensions
            print("Warning: Could not find model.config, using fallback dimensions")
            num_heads = 12
            head_dim = 64
            num_layers = 12
            max_seq_len = 2048

        # =====================================================================
        # SETUP: Create KV cache for the entire generation (if enabled)
        # =====================================================================
        # Create ONE cache that's large enough for prompt + generation
        # This avoids the overhead of creating two caches and copying between them

        kv_cache = None
        if self.use_kv_cache:
            estimated_length = len(prompt_tokens) + self.max_tokens
            if estimated_length > max_seq_len:
                estimated_length = max_seq_len

            kv_cache = KVCache(
                batch_size=1,
                num_heads=num_heads,
                seq_len=estimated_length,
                head_dim=head_dim,
                num_layers=num_layers,
            )

        # =====================================================================
        # PHASE 1: PREFILL - Process entire prompt at once
        # =====================================================================
        # Process all prompt tokens in a single forward pass to initialize
        # the KV cache efficiently.

        # Process all prompt tokens at once
        prompt_tensor = torch.tensor(
            [prompt_tokens], dtype=torch.long, device=self.device
        )

        with torch.amp.autocast(
            device_type=(
                self.device.type if hasattr(self.device, "type") else str(self.device)
            ),
            dtype=torch.bfloat16,
        ):
            # Forward pass - caches K,V for all prompt tokens
            logits, _ = self.model(prompt_tensor, kv_cache=kv_cache)

        # Get logits for the last position
        next_token_logits = logits[0, -1, :]

        # =====================================================================
        # PHASE 2: DECODE - Setup for autoregressive generation
        # =====================================================================
        # Now we generate tokens autoregressively using the same cache

        # =====================================================================
        # STATE INITIALIZATION for tool use
        # =====================================================================
        generated_tokens = list(prompt_tokens)  # All tokens generated so far
        in_python_block = False  # Are we inside <|python|>...<|python_end|>?
        python_expr_tokens = []  # Tokens collected inside python block
        forced_tokens = []  # Queue of tokens to force (calculator output)

        # =====================================================================
        # MAIN GENERATION LOOP with KV caching + tool use
        # =====================================================================
        for _ in range(self.max_tokens):
            # -----------------------------------------------------------------
            # STEP 1: Check if we have forced tokens to inject
            # -----------------------------------------------------------------
            # When calculator returns a result, we force those tokens into
            # generation instead of sampling. We still need to run them through
            # the model to update the KV cache!
            if forced_tokens:
                # Get next forced token
                next_token = forced_tokens.pop(0)
                generated_tokens.append(next_token)

                # Check sequence length limit before running model
                if (
                    hasattr(self.model, "max_seq_len")
                    and len(generated_tokens) >= self.model.max_seq_len
                ):
                    break

                # Run through model to update KV cache (but ignore logits)
                # This ensures the forced tokens are in the cached context
                # When KV cache is disabled, we need to pass the entire sequence
                if self.use_kv_cache:
                    # KV cache enabled: pass only new token
                    next_token_tensor = torch.tensor(
                        [[next_token]], dtype=torch.long, device=self.device
                    )
                else:
                    # No KV cache: pass entire sequence
                    next_token_tensor = torch.tensor(
                        [generated_tokens], dtype=torch.long, device=self.device
                    )

                with torch.amp.autocast(
                    device_type=(
                        self.device.type
                        if hasattr(self.device, "type")
                        else str(self.device)
                    ),
                    dtype=torch.bfloat16,
                ):
                    logits, _ = self.model(
                        next_token_tensor, kv_cache=kv_cache
                    )  # shape (1, 1, vocab_size)

                # Get logits for next iteration
                next_token_logits = logits[0, -1, :]

                # Skip to next iteration
                continue

            # -----------------------------------------------------------------
            # STEP 2: Sample next token from logits
            # -----------------------------------------------------------------
            if self.temperature > 0:
                # Apply temperature scaling
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

                # Sample from probability distribution
                probs = torch.nn.functional.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1).item()
            else:
                # Greedy decoding (deterministic)
                next_token = next_token_logits.argmax().item()

            # -----------------------------------------------------------------
            # STEP 3: Check for termination token
            # -----------------------------------------------------------------
            if next_token == assistant_end_id:
                break

            # -----------------------------------------------------------------
            # STEP 4: Tool-use state machine
            # -----------------------------------------------------------------
            # Track state transitions for calculator tool use

            if next_token == python_start:
                # ENTERING PYTHON MODE
                # Model wants to compute something
                in_python_block = True
                python_expr_tokens = []
                generated_tokens.append(next_token)

            elif next_token == python_end and in_python_block:
                # EXITING PYTHON MODE
                # Execute the collected expression
                in_python_block = False
                generated_tokens.append(next_token)

                # Execute calculator tool
                if python_expr_tokens:
                    expr = self.tokenizer.decode(python_expr_tokens)
                    result = use_calculator(expr)

                    if result is not None:
                        # Convert result to tokens and queue for forcing
                        result_str = str(result)
                        result_tokens = self.tokenizer.encode(result_str)

                        forced_tokens.append(output_start)
                        forced_tokens.extend(result_tokens)
                        forced_tokens.append(output_end)

                python_expr_tokens = []

            elif in_python_block:
                # INSIDE PYTHON MODE
                # Collect tokens for the expression
                python_expr_tokens.append(next_token)
                generated_tokens.append(next_token)

            else:
                # NORMAL MODE
                # Regular text generation
                generated_tokens.append(next_token)

            # -----------------------------------------------------------------
            # STEP 5: Check sequence length limits
            # -----------------------------------------------------------------
            if (
                hasattr(self.model, "max_seq_len")
                and len(generated_tokens) >= self.model.max_seq_len
            ):
                break

            # -----------------------------------------------------------------
            # STEP 6: Get logits for next token (KEY KV CACHE OPTIMIZATION)
            # -----------------------------------------------------------------
            # With KV cache: Pass only the ONE new token, reusing all cached K,V.
            # Without KV cache: Pass entire sequence each time (slower).

            if self.use_kv_cache:
                # KV cache enabled: pass only new token
                next_token_tensor = torch.tensor(
                    [[next_token]], dtype=torch.long, device=self.device
                )
            else:
                # No KV cache: pass entire sequence
                next_token_tensor = torch.tensor(
                    [generated_tokens], dtype=torch.long, device=self.device
                )

            with torch.amp.autocast(
                device_type=(
                    self.device.type
                    if hasattr(self.device, "type")
                    else str(self.device)
                ),
                dtype=torch.bfloat16,
            ):
                # Forward pass with or without KV cache
                logits, _ = self.model(next_token_tensor, kv_cache=kv_cache)

            # Extract logits for next prediction
            next_token_logits = logits[0, -1, :]

        # =====================================================================
        # FINALIZATION: Decode and return generated text
        # =====================================================================
        # Extract only newly generated tokens (exclude the prompt)
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

            # Generate completion (with tool use if supported)
            try:
                if self.supports_tools:
                    generated_text = self.generate_completion_with_tools(prompt_tokens)
                else:
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
