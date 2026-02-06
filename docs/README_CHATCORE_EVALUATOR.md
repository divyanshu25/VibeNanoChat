# ChatCORE Evaluator

A generative evaluation system for testing language models on chat-based tasks during training. Unlike likelihood-based benchmarks (like CORE), ChatCORE evaluates models by generating actual text completions and checking them for correctness.

## Table of Contents

- [Overview](#overview)
- [Quick Start](#quick-start)
- [Supported Tasks](#supported-tasks)
- [Key Features](#key-features)
- [KV Caching Optimization](#kv-caching-optimization)
- [Tool Use (Calculator)](#tool-use-calculator)
- [Usage Examples](#usage-examples)
- [API Reference](#api-reference)
- [Distributed Evaluation](#distributed-evaluation)
- [Integration with Training](#integration-with-training)
- [Troubleshooting](#troubleshooting)

---

## Overview

ChatCORE (Chat-based Comprehensive Reasoning Evaluation) is designed for evaluating chat-optimized language models on tasks that require text generation:

- **GSM8K**: Grade school math reasoning problems
- **HumanEval**: Python code generation with test execution
- *Future*: MMLU, ARC, SpellingBee

### Why ChatCORE?

Traditional benchmarks use multiple-choice questions and measure likelihood scores. ChatCORE is more realistic:
- Models generate free-form text (like in real usage)
- Supports tool use (calculator for math)
- Optimized with KV caching for 5-10x speedup
- Integrates seamlessly into training loops

---

## Quick Start

### 5-Minute Example

```python
from eval_tasks.chat_core.evaluator import ChatCoreEvaluator
from eval_tasks.chat_core.gsm8k import load_gsm8k, evaluate_gsm8k, render_gsm8k_prompt

# Step 1: Initialize evaluator
evaluator = ChatCoreEvaluator(
    model=my_model,
    tokenizer=my_tokenizer,
    device="cuda",
    master_process=True,
    max_examples=100,        # Limit examples per task
    temperature=0.0,         # Greedy decoding
    max_tokens=512,
    use_kv_cache=True,       # 5-10x speedup
)

# Step 2: Register task
evaluator.register_task("GSM8K", {
    'load_fn': load_gsm8k,
    'eval_fn': evaluate_gsm8k,
    'render_fn': render_gsm8k_prompt,
})

# Step 3: Run evaluation
results = evaluator.evaluate_all_tasks(step=5000, global_step=5000)

# Output:
# GSM8K Accuracy: 0.6800 (68%)
```

---

## Supported Tasks

### GSM8K (Grade School Math 8K)

**What it tests**: Multi-step mathematical reasoning

**Example problem**:
```
Question: Janet's ducks lay 16 eggs per day. She eats three for breakfast 
every morning and bakes muffins for her friends every day with four. She 
sells the remainder at the farmers' market daily for $2 per fresh duck egg. 
How much in dollars does she make every day at the farmers' market?

Expected answer: 18
```

**Model generation with calculator**:
```
Janet has 16 eggs.
She uses <|python|>3+4<|python_end|><|output_start|>7<|output_end|> eggs.
She has <|python|>16-7<|python_end|><|output_start|>9<|output_end|> eggs left.
She makes <|python|>9*2<|python_end|><|output_start|>18<|output_end|> dollars.
The answer is 18.
```

**Evaluation**: Extracts the final number and compares to ground truth.

### HumanEval

**What it tests**: Python code generation and correctness

**Example problem**:
```python
def has_close_elements(numbers: List[float], threshold: float) -> bool:
    """ Check if in given list of numbers, are any two numbers closer to each other than
    given threshold.
    >>> has_close_elements([1.0, 2.0, 3.0], 0.5)
    False
    >>> has_close_elements([1.0, 2.8, 3.0, 4.0, 5.0, 2.0], 0.3)
    True
    """
```

**Model generation**:
```python
    for i in range(len(numbers)):
        for j in range(i + 1, len(numbers)):
            if abs(numbers[i] - numbers[j]) < threshold:
                return True
    return False
```

**Evaluation**: Executes the generated code against test cases with timeouts and safety checks.

---

## Key Features

### 1. **KV Caching Optimization**

Speeds up generation by 5-10x by caching attention keys and values. See [KV Caching Optimization](#kv-caching-optimization) for details.

### 2. **Tool Use Support**

Models can use a calculator for accurate arithmetic:
```
Model: "The answer is <|python|>12.5 * 0.15<|python_end|>"
→ Calculator: 1.875
→ Forced: "<|output_start|>1.875<|output_end|>"
```

### 3. **Distributed Evaluation**

Automatically splits work across multiple GPUs with `DistributedDataParallel`.

### 4. **Detailed Logging**

First 5 examples show full prompts, generations, and evaluation details for debugging.

### 5. **Wandb Integration**

Automatically logs metrics:
- `chatcore/GSM8K`: Individual task accuracy
- `chatcore/HumanEval`: Individual task accuracy  
- `chatcore_score`: Average across all tasks

---

## KV Caching Optimization

### The Problem: O(N²) Complexity

Without caching, each generation step reprocesses all previous tokens:

```
Step 1: Process [token1] → predict token2
Step 2: Process [token1, token2] → predict token3  (recomputes token1!)
Step 3: Process [token1, token2, token3] → predict token4  (recomputes all!)
```

For a 512-token generation, this means ~131,000 token processing operations!

### The Solution: Two-Phase Generation

**Phase 1: PREFILL** (Process prompt once)
```python
# Process entire prompt at once, store K/V for each layer
prompt_tokens = [15, 496, 468, ...]  # "What is 2+2?"
next_token_logits = prefill_prompt(model, prompt_tokens, kv_cache, device)
```

**Phase 2: DECODE** (Process only new tokens)
```python
for _ in range(max_tokens):
    next_token = sample_next_token(next_token_logits, temperature, top_k)
    
    # Only process the new token (cache has rest)
    next_token_logits = forward_pass(model, next_token, kv_cache, device)
```

### Performance Gains

| Sequence Length | Without Cache | With Cache | Speedup |
|----------------|---------------|------------|---------|
| 128 tokens     | 8,256 ops     | 128 ops    | 64x     |
| 256 tokens     | 32,896 ops    | 256 ops    | 128x    |
| 512 tokens     | 131,584 ops   | 512 ops    | 257x    |

Real-world speedup: **5-10x** (due to overhead and implementation)

### Implementation Details

```python
# Create KV cache (from kv_cache_utils.py)
kv_cache = create_kv_cache(
    prompt_len=len(prompt_tokens),
    max_new_tokens=max_tokens,
    num_heads=model.config.n_head,
    head_dim=model.config.n_embd // model.config.n_head,
    num_layers=model.config.n_layer,
    max_seq_len=model.config.block_size,
    use_cache=True
)

# Cache structure:
# kv_cache[layer_idx] = {
#     'k': tensor of shape [batch=1, num_heads, seq_len, head_dim],
#     'v': tensor of shape [batch=1, num_heads, seq_len, head_dim],
#     'pos': current position (int)
# }
```

---

## Tool Use (Calculator)

### Overview

Models can execute Python expressions during generation for accurate math. This is critical for GSM8K where floating-point arithmetic matters.

### Special Tokens

Your tokenizer must support these special tokens:
- `<|python|>`: Start of Python expression
- `<|python_end|>`: End of Python expression
- `<|output_start|>`: Start of calculator result
- `<|output_end|>`: End of calculator result

### State Machine

The generator implements a state machine that tracks tool use:

```python
in_python_block = False      # Are we collecting Python code?
python_expr_tokens = []      # Accumulated expression tokens
forced_tokens = []           # Calculator results to inject
```

### Generation Flow

```
1. Model generates: "The price is "
   → State: normal generation

2. Model generates: "<|python|>"
   → State: in_python_block = True
   → Start collecting expression

3. Model generates: "12.99", " * ", "0.15"
   → State: still in python block
   → Accumulate: python_expr_tokens = [12.99, *, 0.15]

4. Model generates: "<|python_end|>"
   → State: in_python_block = False
   → Execute: use_calculator("12.99 * 0.15") = 1.9485
   → Queue forced tokens: ["<|output_start|>", "1", ".", "9485", "<|output_end|>"]

5. System injects forced tokens
   → Generation continues with: "<|output_start|>1.9485<|output_end|>"

6. Model sees the result and continues: " dollars"
   → State: normal generation
```

### Example Generation

**Prompt**: "Janet's ducks lay 16 eggs per day. She eats 3 for breakfast and bakes 4 into muffins. How many eggs remain?"

**Model output**:
```
Janet starts with 16 eggs.
She uses <|python|>3 + 4<|python_end|><|output_start|>7<|output_end|> eggs.
She has <|python|>16 - 7<|python_end|><|output_start|>9<|output_end|> eggs remaining.
The answer is 9.
```

### Calculator Implementation

```python
from .tools import use_calculator

# Safe execution with sympy
result = use_calculator("12.99 * 0.15")
# Returns: 1.9485

# Handles edge cases:
use_calculator("1/0")          # Returns None (division by zero)
use_calculator("import os")    # Returns None (blocked imports)
use_calculator("sqrt(2)")      # Returns 1.41421... (sympy functions)
```

### Checking Tool Support

```python
# The evaluator automatically checks on initialization
if hasattr(tokenizer, "_special_tokens"):
    python_start = tokenizer._special_tokens.get("<|python|>")
    if python_start is not None:
        self.supports_tools = True
```

If tool tokens aren't available, the evaluator falls back to `generate_completion()` (no calculator).

---

## Usage Examples

### Example 1: Basic Evaluation

```python
from eval_tasks.chat_core.evaluator import ChatCoreEvaluator
from eval_tasks.chat_core.gsm8k import load_gsm8k, evaluate_gsm8k, render_gsm8k_prompt

evaluator = ChatCoreEvaluator(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    master_process=True,
)

evaluator.register_task("GSM8K", {
    'load_fn': load_gsm8k,
    'eval_fn': evaluate_gsm8k,
    'render_fn': render_gsm8k_prompt,
})

results = evaluator.evaluate_task("GSM8K")
print(f"Accuracy: {results['accuracy']:.2%}")
# Output: Accuracy: 68.00%
```

### Example 2: Multiple Tasks

```python
from eval_tasks.chat_core.gsm8k import load_gsm8k, evaluate_gsm8k, render_gsm8k_prompt
from eval_tasks.chat_core.humaneval import load_humaneval, evaluate_humaneval, render_humaneval_prompt

evaluator = ChatCoreEvaluator(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    master_process=True,
    max_examples=50,  # Limit to 50 examples per task
)

# Register multiple tasks
evaluator.register_task("GSM8K", {
    'load_fn': load_gsm8k,
    'eval_fn': evaluate_gsm8k,
    'render_fn': render_gsm8k_prompt,
})

evaluator.register_task("HumanEval", {
    'load_fn': load_humaneval,
    'eval_fn': evaluate_humaneval,
    'render_fn': render_humaneval_prompt,
})

# Evaluate all tasks at once
all_results = evaluator.evaluate_all_tasks(global_step=10000)

# Output:
# GSM8K: 0.6800 (82.5s)
# HumanEval: 0.7700 (62.8s)
# ChatCORE Score: 0.7250
```

### Example 3: Temperature Sampling

```python
# Greedy decoding (deterministic)
evaluator = ChatCoreEvaluator(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    master_process=True,
    temperature=0.0,  # Always pick most likely token
)

# Creative sampling (stochastic)
evaluator = ChatCoreEvaluator(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    master_process=True,
    temperature=0.8,  # More randomness
    top_k=50,         # Consider top 50 tokens
    num_samples=5,    # Generate 5 samples per problem (for pass@5)
)
```

### Example 4: Distributed Multi-GPU Evaluation

```python
import torch.distributed as dist

# Initialize DDP
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

evaluator = ChatCoreEvaluator(
    model=model,
    tokenizer=tokenizer,
    device=f"cuda:{rank}",
    master_process=(rank == 0),
    ddp=True,
    ddp_rank=rank,
    ddp_world_size=world_size,
)

evaluator.register_task("GSM8K", {...})

# Each GPU processes every world_size-th example
# Results are automatically aggregated
results = evaluator.evaluate_all_tasks(global_step=5000)
```

---

## API Reference

### `ChatCoreEvaluator`

#### `__init__()`

```python
ChatCoreEvaluator(
    model,                          # Model with forward() method
    tokenizer,                      # Tokenizer with encode()/decode()
    device,                         # torch device ("cuda", "cpu")
    master_process: bool,           # Whether to print/log (use rank==0)
    ddp: bool = False,             # Using DistributedDataParallel?
    ddp_rank: int = 0,             # Current process rank
    ddp_world_size: int = 1,       # Total number of processes
    max_examples: Optional[int] = None,  # Limit examples per task
    num_samples: int = 1,          # Samples per problem (for pass@k)
    max_tokens: int = 512,         # Max tokens to generate
    temperature: float = 0.0,      # Sampling temperature (0=greedy)
    top_k: int = 50,               # Top-k sampling
    use_kv_cache: bool = True,     # Enable KV caching (5-10x faster)
)
```

#### `register_task()`

```python
evaluator.register_task(
    task_name: str,       # Name like "GSM8K"
    task_config: Dict     # Config with load_fn, eval_fn, render_fn
)
```

**task_config structure**:
```python
{
    'load_fn': callable,    # () -> List[Dict]
                           # Returns list of examples
    
    'eval_fn': callable,    # (example, generated_text) -> bool
                           # Or (example, generated_text, return_details=True) -> Dict
                           # Returns True/False or detailed results
    
    'render_fn': callable,  # (example) -> List[int]
                           # Returns tokenized prompt
}
```

#### `evaluate_task()`

```python
results = evaluator.evaluate_task(task_name: str)
```

**Returns**:
```python
{
    'accuracy': float,      # 0.0 to 1.0
    'correct': int,         # Number of correct answers
    'total': int,           # Number of examples evaluated
    'num_evaluated': int,   # Same as total
}
```

#### `evaluate_all_tasks()`

```python
all_results = evaluator.evaluate_all_tasks(
    step: Optional[int] = None,         # Current step within epoch
    global_step: Optional[int] = None,  # Global step across epochs
)
```

**Returns**:
```python
{
    'GSM8K': {
        'accuracy': 0.68,
        'correct': 680,
        'total': 1000,
        'num_evaluated': 1000
    },
    'HumanEval': {
        'accuracy': 0.77,
        'correct': 77,
        'total': 100,
        'num_evaluated': 100
    }
}
```

#### `generate_completion()`

```python
generated_text = evaluator.generate_completion(
    prompt_tokens: List[int]  # Tokenized prompt
)
```

Basic generation without tool use. Uses KV caching if enabled.

#### `generate_completion_with_tools()`

```python
generated_text = evaluator.generate_completion_with_tools(
    prompt_tokens: List[int]  # Tokenized prompt
)
```

Generation with calculator support. Falls back to `generate_completion()` if tool tokens unavailable.

---

## Distributed Evaluation

### How It Works

Each GPU processes a disjoint subset of examples:

```python
# GPU 0 processes: examples 0, 4, 8, 12, ...
# GPU 1 processes: examples 1, 5, 9, 13, ...
# GPU 2 processes: examples 2, 6, 10, 14, ...
# GPU 3 processes: examples 3, 7, 11, 15, ...

for idx in range(ddp_rank, num_examples, ddp_world_size):
    example = data[idx]
    # ... evaluate example ...
```

Results are aggregated using `torch.distributed.all_reduce()`:

```python
# Each GPU has local counts
# GPU 0: correct=170, total=250
# GPU 1: correct=165, total=250
# GPU 2: correct=168, total=250
# GPU 3: correct=172, total=250

dist.all_reduce(correct_tensor, op=dist.ReduceOp.SUM)
dist.all_reduce(total_tensor, op=dist.ReduceOp.SUM)

# Final: correct=675, total=1000
# Accuracy: 0.675
```

### Setup Example

```python
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP

# Initialize process group
dist.init_process_group(backend='nccl')
rank = dist.get_rank()
world_size = dist.get_world_size()

# Wrap model in DDP
model = DDP(model, device_ids=[rank])

# Create evaluator with DDP settings
evaluator = ChatCoreEvaluator(
    model=model,
    tokenizer=tokenizer,
    device=f"cuda:{rank}",
    master_process=(rank == 0),  # Only rank 0 prints/logs
    ddp=True,
    ddp_rank=rank,
    ddp_world_size=world_size,
)
```

---

## Integration with Training

### Periodic Evaluation

```python
from eval_tasks.chat_core.evaluator import ChatCoreEvaluator
from eval_tasks.chat_core.gsm8k import load_gsm8k, evaluate_gsm8k, render_gsm8k_prompt

# Setup (once at start of training)
evaluator = ChatCoreEvaluator(
    model=model,
    tokenizer=tokenizer,
    device="cuda",
    master_process=(rank == 0),
    max_examples=100,  # Faster evaluation
    use_kv_cache=True,
)

evaluator.register_task("GSM8K", {
    'load_fn': load_gsm8k,
    'eval_fn': evaluate_gsm8k,
    'render_fn': render_gsm8k_prompt,
})

# Training loop
for step in range(max_steps):
    # Training step
    loss = train_step(model, batch)
    
    # Evaluate every 1000 steps
    if step % 1000 == 0 and step > 0:
        model.eval()
        results = evaluator.evaluate_all_tasks(global_step=step)
        model.train()
        
        # Results automatically logged to wandb
        # chatcore/GSM8K, chatcore_score, etc.
```

### Example Output During Training

```
Step 0: loss=3.456
Step 100: loss=2.987
...
Step 1000: loss=1.234

================================================================================
💬 ChatCORE EVALUATION | Step 1000
================================================================================

Evaluating GSM8K: 100 examples
================================================================================
📝 Example 1/100
================================================================================
🔵 PROMPT:
Question: Janet has 12 apples...
================================================================================

🤖 MODEL GENERATED:
Janet eats <|python|>12*0.25<|python_end|><|output_start|>3.0<|output_end|> apples...
================================================================================

✅ CORRECT
================================================================================
  Progress: 10/100 | Accuracy: 0.700
  Progress: 20/100 | Accuracy: 0.650
  ...
  Progress: 100/100 | Accuracy: 0.680

  ✓ GSM8K Accuracy: 0.6800 (680/1000)
  GSM8K: 0.6800 (45.2s)

================================================================================
💬 ChatCORE EVALUATION | Step 1000
   Average Score: 0.6800
   Total Time: 45.20s
================================================================================

Step 1001: loss=1.221
...
```

---

## Troubleshooting

### Problem: Tool use not working

**Symptoms**: Model doesn't use `<|python|>` tags, or evaluator says "Tool use enabled: False"

**Solution**: Check tokenizer has required special tokens:
```python
# Check if tokens exist
required_tokens = ["<|python|>", "<|python_end|>", "<|output_start|>", "<|output_end|>"]
for token in required_tokens:
    if token not in tokenizer._special_tokens:
        print(f"Missing token: {token}")
```

Add tokens to tokenizer:
```python
tokenizer.add_special_tokens({
    "<|python|>": 50260,
    "<|python_end|>": 50261,
    "<|output_start|>": 50262,
    "<|output_end|>": 50263,
})
```

### Problem: Slow generation

**Symptoms**: Evaluation takes hours

**Solutions**:
1. **Enable KV caching**: `use_kv_cache=True` (5-10x speedup)
2. **Reduce max_tokens**: Lower `max_tokens=256` instead of 512
3. **Limit examples**: Use `max_examples=50` for faster testing
4. **Use multiple GPUs**: Distribute with DDP

### Problem: OOM (Out of Memory)

**Symptoms**: CUDA out of memory errors during evaluation

**Solutions**:
1. **Reduce max_tokens**: Lower to 256 or 128
2. **Reduce batch size**: Evaluator uses batch_size=1, but check model internals
3. **Clear cache**: Call `torch.cuda.empty_cache()` between evaluations
4. **Smaller model**: Use fewer layers/parameters

```python
# Add cache clearing
def evaluate_with_cache_clearing():
    results = evaluator.evaluate_task("GSM8K")
    torch.cuda.empty_cache()
    return results
```

### Problem: Inaccurate evaluation

**Symptoms**: Accuracy seems wrong, or model answers are correct but marked incorrect

**Solutions**:
1. **Check answer extraction**: Verify `eval_fn` extracts answers correctly
2. **Review first 5 examples**: Evaluator prints detailed output for debugging
3. **Temperature too high**: Use `temperature=0.0` for deterministic greedy decoding
4. **Check prompt format**: Ensure `render_fn` produces correct prompts

```python
# Debug by checking what the model sees
evaluator = ChatCoreEvaluator(..., temperature=0.0)  # Deterministic
results = evaluator.evaluate_task("GSM8K")

# First 5 examples will show:
# - Decoded prompt
# - Model generation
# - Expected vs predicted answer
# - Whether match succeeded
```

### Problem: DDP hanging or deadlock

**Symptoms**: Evaluation hangs when using multiple GPUs

**Solutions**:
1. **Check all ranks**: Ensure all processes call evaluation simultaneously
2. **Barrier before reduce**: Already handled by evaluator (line 604)
3. **Match example counts**: All ranks must process same total examples

```python
# Make sure all ranks reach evaluation
if step % 1000 == 0:
    # This runs on ALL ranks
    results = evaluator.evaluate_all_tasks(global_step=step)
```

### Problem: Wandb not logging

**Symptoms**: Metrics don't appear in wandb dashboard

**Solutions**:
1. **Initialize wandb first**:
```python
import wandb
wandb.init(project="my-project", name="run-1")

# Then create evaluator
evaluator = ChatCoreEvaluator(...)
```

2. **Check master_process**: Only rank 0 logs to wandb
```python
evaluator = ChatCoreEvaluator(
    ...,
    master_process=(rank == 0),  # Critical!
)
```

3. **Check exceptions**: Evaluator silently catches wandb errors (lines 665-667)

---

## Performance Tips

### 1. Use KV Caching (5-10x speedup)

```python
evaluator = ChatCoreEvaluator(..., use_kv_cache=True)
```

### 2. Limit Examples for Fast Iteration

```python
# During development
evaluator = ChatCoreEvaluator(..., max_examples=20)

# Final evaluation
evaluator = ChatCoreEvaluator(..., max_examples=None)  # All examples
```

### 3. Use Multiple GPUs

```python
# 4 GPUs = 4x speedup
evaluator = ChatCoreEvaluator(
    ...,
    ddp=True,
    ddp_rank=rank,
    ddp_world_size=4,
)
```

### 4. Optimize max_tokens

```python
# Most GSM8K answers < 300 tokens
evaluator = ChatCoreEvaluator(..., max_tokens=300)

# HumanEval needs more for code
evaluator = ChatCoreEvaluator(..., max_tokens=512)
```

### 5. Profile Your Evaluation

```python
import time

start = time.time()
results = evaluator.evaluate_task("GSM8K")
elapsed = time.time() - start

examples_per_sec = results['total'] / elapsed
print(f"Throughput: {examples_per_sec:.2f} examples/sec")
```

---

## Files

### Core Files
- `evaluator.py`: Main evaluator class (this document)
- `kv_cache_utils.py`: KV caching implementation
- `tools.py`: Calculator and tool use utilities

### Task Files
- `gsm8k.py`: GSM8K task implementation
- `humaneval.py`: HumanEval task implementation
- `arc_challenge.py`: ARC-Challenge task (future)

### Utility Files
- `utils.py`: Shared utilities (prompt rendering, etc.)

---

## Citation

If you use ChatCORE evaluator in your research, please cite:

```bibtex
@software{chatcore_evaluator,
  title = {ChatCORE: Generative Evaluation for Chat Models},
  author = {VibeNanoChat Team},
  year = {2026},
  url = {https://github.com/your-repo/VibeNanoChat}
}
```

---

## License

[Your License Here]

---

## Contributing

Contributions welcome! To add a new task:

1. Create task file (e.g., `my_task.py`)
2. Implement three functions:
   - `load_my_task()`: Returns `List[Dict]` of examples
   - `evaluate_my_task(example, generated_text)`: Returns `bool` or `Dict`
   - `render_my_task_prompt(example)`: Returns `List[int]` token IDs
3. Register in your training script:
   ```python
   evaluator.register_task("MyTask", {
       'load_fn': load_my_task,
       'eval_fn': evaluate_my_task,
       'render_fn': render_my_task_prompt,
   })
   ```

See `gsm8k.py` for a complete example.
