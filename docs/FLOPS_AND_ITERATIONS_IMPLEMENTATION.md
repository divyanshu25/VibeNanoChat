# FLOPs and Iterations Calculation Implementation

## Overview

This implementation adds nanochat-style FLOPs estimation and automatic iteration calculation to NanoGPT, allowing you to specify training horizons based on computational budget or data:param ratios.

## What Was Implemented

### 1. FLOPs Estimation (`estimate_flops()` in `gpt2_model.py`)

Calculates the FLOPs per token for training (forward + backward pass):

```python
num_flops_per_token = 6 * (matmul_params) + 12 * n_head * head_dim * seq_len * n_layer
```

**Formula breakdown:**
- **6 FLOPs per matmul parameter**: 2 (forward) + 4 (backward) = 6
- **Attention FLOPs**: 12 * n_head * head_dim * seq_len per layer
- **Excludes**: Embeddings (lookup, not matmul) and layer norms (negligible)

**References:**
- [The FLOPs Calculus of Language Model Training](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4)
- [PaLM Paper (Appendix B)](https://arxiv.org/abs/2204.02311)

### 2. Scaling Parameters Count (`num_scaling_params()` in `gpt2_model.py`)

Returns total number of model parameters for scaling law calculations.

Follows Chinchilla paper approach: includes all parameters.
- **Ref**: [Chinchilla Paper](https://arxiv.org/abs/2203.15556)

### 3. Iteration Calculation (`calculate_num_iterations()` in `utils.py`)

Automatically calculates training iterations based on three possible specifications (priority order):

#### Priority 1: Explicit Iterations
```python
config.num_iterations = 10000  # Train for exactly 10K steps
```

#### Priority 2: Target FLOPs
```python
config.target_flops = 1e20  # Train until 10^20 FLOPs
# Iterations = target_flops / (flops_per_token * batch_size)
```

#### Priority 3: Data:Param Ratio (Default)
```python
config.target_param_data_ratio = 20  # Chinchilla optimal
# target_tokens = ratio * num_params
# Iterations = target_tokens / batch_size
```

### 4. Configuration Parameters (`config.py`)

Added to `GPTConfig`:
```python
# Training horizon calculation
num_iterations: int = -1              # -1 = auto-calculate
target_flops: float = -1.0            # -1 = don't use
target_param_data_ratio: int = 20     # Chinchilla optimal ratio
```

## Verification with Nanochat 560M Model

### Test Results

Running with nanochat's 560M configuration:
- **vocab_size**: 65,536
- **n_layer**: 20  
- **n_embed**: 1,280
- **n_head**: 10
- **block_size**: 2,048

### FLOPs Calculation
```
Our calculation:    3.493140e+09
Nanochat output:    3.491758e+09
Difference:         0.04% ✓
```
✅ **Extremely accurate match!**

### Iteration Calculation (ratio=20)
```
Our calculation:    21,512 iterations
Nanochat output:    21,400 iterations
Parameters (ours):  563,944,960
Parameters (nano):  560,988,160
```

Small difference due to:
- Our model uses standard GPT-2 architecture
- Nanochat has custom components (resid_lambdas, x0_lambdas)
- ~3M parameter difference leads to ~100 iteration difference

✅ **Calculation logic matches exactly!**

## Usage Examples

### Example 1: Chinchilla Optimal Training (20:1 ratio)
```python
from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT
from gpt_2.utils import calculate_num_iterations

config = GPTConfig()
config.target_param_data_ratio = 20  # Default
config.total_batch_size = 524288

model = GPT(config)

# Automatically calculates iterations
num_iterations, flops_per_token, num_params = calculate_num_iterations(model, config)

print(f"Will train for {num_iterations:,} iterations")
print(f"FLOPs per token: {flops_per_token:e}")
print(f"Total tokens: {num_iterations * config.total_batch_size:,}")
```

Output:
```
Calculated number of iterations from target data:param ratio: 21,512
Estimated FLOPs per token: 3.493140e+09
Total number of training tokens: 11,278,483,456
Tokens : Params ratio: 20.00
Total training FLOPs estimate: 3.939733e+19
Will train for 21,512 iterations
```

### Example 2: Fixed Computational Budget
```python
config = GPTConfig()
config.num_iterations = -1           # Disable explicit
config.target_flops = 1e20           # 10^20 FLOPs budget
config.target_param_data_ratio = -1  # Disable ratio

num_iterations, _, _ = calculate_num_iterations(model, config)
```

### Example 3: Explicit Iteration Count
```python
config = GPTConfig()
config.num_iterations = 10000        # Train for exactly 10K steps

num_iterations, _, _ = calculate_num_iterations(model, config)
# Returns: 10000
```

## Key Features

✅ **Accurate FLOPs Estimation**: Within 0.04% of nanochat  
✅ **Flexible Training Horizons**: Iterations, FLOPs, or data:param ratio  
✅ **Chinchilla Optimal by Default**: ratio=20  
✅ **Priority System**: Clear precedence when multiple options set  
✅ **Detailed Statistics**: Prints training metrics automatically  
✅ **Nanochat Compatible**: Same calculation logic as reference implementation  

## Integration with Trainer

To use in training scripts:

```python
from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT
from gpt_2.utils import calculate_num_iterations

# Create model and config
config = GPTConfig()
model = GPT(config)

# Calculate iterations automatically
num_iterations, flops_per_token, num_params = calculate_num_iterations(model, config)

# Pass to trainer or use in training loop
config.num_epochs = 1  # Nanochat uses iterations, not epochs
max_steps = num_iterations
```

## Scaling Law Recommendations

### Data:Param Ratios
- **Chinchilla Optimal**: 20:1 (default)
- **Underfitting regime**: 5:1 to 10:1
- **Overfitting regime**: >30:1
- **Nanochat experiments**: Often use 4:1 for faster iteration

### FLOPs-Based Training
- Small models (~100M): 1e18 - 1e19 FLOPs
- Medium models (~500M): 1e19 - 5e19 FLOPs  
- Large models (~1B+): 5e19 - 1e21 FLOPs

## References

1. **FLOPs Calculation**: [Bahdanau (Medium)](https://medium.com/@dzmitrybahdanau/the-flops-calculus-of-language-model-training-3b19c1f025e4)
2. **PaLM FLOPs Formula**: [Chowdhery et al. 2022](https://arxiv.org/abs/2204.02311)
3. **Chinchilla Scaling Laws**: [Hoffmann et al. 2022](https://arxiv.org/abs/2203.15556)
4. **Nanochat Implementation**: [kellyjordan/nanochat](https://github.com/kellyjordan/nanochat)
