# VibeNanoChat

> **Standing on the shoulders of giants**: This repo is essentially Andrej Karpathy's excellent [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat) combined together with some organizational changes. All the good ideas (architecture, depth parameterization, training setup, optimizers, scaling laws) come directly from Andrej's work. This is a fork/reorganization for our own experiments. **All credit goes to @karpathy!** 🙏

## Why this repo?

This repo is a reorganization attempt at the original nanoGPT and nanochat codebases. The goal is to bring a cleaner code structure with detailed comments to help beginners learn and not get lost in the code. While all the core ideas and implementation come from Andrej Karpathy's work, we've tried to make it more approachable for those new to transformer training.

## What you get

- 🚀 Train GPT-2 (124M) 
- 🎯 One-knob scaling: `DEPTH=12` controls everything, LR/WD auto-scale
- 📊 Built-in evaluation: 35+ benchmarks (MMLU, HellaSwag, etc.)
- 💬 Chat with your model: Web UI included
- 🔬 Scaling law tools: Run experiments and plot Chinchilla-style curves
- 🌐 Modern datasets: FineWeb-Edu (quality!), TaskMixture for instruction tuning

## Quick start: Train your first GPT

```bash
# Install
make environment

# Get some data
cd data/fineweb_edu && uv run python prepare_parquet.py --config sample-10BT && cd ../..

# Train
make ddp-train NGPUS=8 MODE=pretraining

# Chat with it
uv run python scripts/chat.py --checkpoint logs/pretraining/step_19531.pt
```

That's it. You now have a GPT that understands language.

## What's in this repo?

**Everything is from Andrej Karpathy's work:**

**From nanoGPT**:
- The entire GPT-2 architecture and training loop
- Clean, minimal code philosophy
- Smart defaults that just work
- DistMuon optimizer with ZeRO-2 style sharding

**From nanochat**:
- Depth parameterization (the `DEPTH` knob that scales everything)
- Scaling law experimental setup
- Auto-scaling of learning rate and weight decay
- Isoflop curve analysis and plotting

**Our changes** (minimal):
- Combined nanoGPT + nanochat into one repo
- Added Makefile helpers for convenience
- Some documentation and code organization

**Bottom line**: This is Andrej's code. We're just using it for our experiments and made it easier to work with for our workflow. All the clever ideas and hard work are his.

## The Depth Knob 🎛️

This is Andrej's genius idea from nanochat: control model size with one parameter.

```bash
# Small model
make ddp-train DEPTH=6 TARGET_FLOPS=1e18

# Medium model (roughly GPT-2 size)
make ddp-train DEPTH=12 TARGET_FLOPS=1e18

# Large model
make ddp-train DEPTH=20 TARGET_FLOPS=1e18
```

The `DEPTH` parameter controls both depth and width. Learning rate and weight decay automatically scale based on model size. No more hyperparameter sweeps!

Want to run a full scaling law experiment? Easy:

```bash
make run-scaling-law  # Sweeps depths 6-14, FLOP budgets 1e18, 2e18, 3e18, 6e18
uv run python scripts/plot_isoflop_curve.py  # Plot results
```

You'll get beautiful Chinchilla-style plots showing optimal model size for your compute budget.

📖 **More details**: [docs/README_DEPTH_PARAMETERIZATION.md](docs/README_DEPTH_PARAMETERIZATION.md)

## Scaling Laws: The Bitter Lesson Applies Here Too 📈

We ran a bunch of experiments sweeping model sizes and FLOP budgets to find the optimal allocation of compute. Here's what we learned.

### The Fundamental Equation: C = 6ND

When you train a transformer, the compute cost is approximately:
```
C = 6ND
```

where:
- `C` = total training FLOPs
- `N` = number of model parameters
- `D` = number of training tokens

The `6` comes from the forward pass (2ND FLOPs) + backward pass (4ND FLOPs). So if you have a fixed compute budget C, you face a tradeoff: bigger model with less data, or smaller model with more data?

#### Does C = 6ND Actually Hold?

Short answer: **yes, remarkably well**. We verified this across 46+ training runs spanning four compute budgets (1e18, 2e18, 3e18, 6e18 FLOPs) and model sizes from 77M to 522M parameters.

<p align="center">
  <img src="scripts/graphs/flops_formula_verification.png" width="100%"/>
</p>

*Figure: Empirical verification of C = 6ND. Left: ratio of actual FLOPs to 6ND across model depths. Right: average ratio per budget. The formula holds to within ±1% on average.*

**Key findings:**

1. **Average accuracy**: The ratio C_actual / (6ND) is ~1.001 across all runs. That's **~0.1% error** - basically perfect.

2. **Consistent across budgets**: 
   - 1e18 FLOPs: 0.998× (−0.24%)
   - 2e18 FLOPs: 0.999× (−0.12%)
   - 3e18 FLOPs: 0.998× (−0.23%)
   - 6e18 FLOPs: 1.009× (+0.94%)

3. **Model size effects**: Small models (N8, N9) slightly underestimate (0.88-0.92×), large models (N15-N20) slightly overestimate (1.04-1.08×). Medium models are spot on.

Why the small deviations? 

- **Small models**: Less optimizer overhead relative to forward/backward. Also, some ops (embeddings, layer norms) don't scale exactly as ND.

- **Large models**: More gradient communication in distributed training, more memory ops, slightly higher overhead from checkpointing and evals.

- **The "6" is approximate**: It assumes every operation in the transformer scales as ND. In reality, some ops scale differently (e.g., softmax is O(seq_len²), not O(params)).

But here's the thing: **these effects nearly cancel out**. The average is so close to 1.0 that C = 6ND is empirically a *great* predictor of training cost. You can use it to plan experiments with confidence.

**Practical implication**: Want to know how many tokens you can afford with your compute budget? Just solve for D:

```
D = C / (6N)
```

Example: You have 1e19 FLOPs and want to train a 500M parameter model. How many tokens can you train?

```
D = 1e19 / (6 × 500M) = 3.33B tokens
```

The formula works. Use it.

### The Square Root Law: N, D ∝ √C

Here's the interesting part. When we fit power laws to our optimal models across four compute budgets (1e18, 2e18, 3e18, 6e18 FLOPs), we found:

```
N_optimal ∝ C^0.456
D_optimal ∝ C^0.544
```

Both exponents are close to 0.5! This means **as you scale compute, you should scale both model size AND training data roughly proportional to the square root of compute**. Not linearly - *sublinearly*. 

Why? Because both N and D contribute to loss reduction, but with diminishing returns. The math works out such that you want to grow them at similar rates.

<p align="center">
  <img src="scripts/graphs/optimal_model_vs_flops.png" width="100%"/>
</p>

*Figure: Log-log plot showing optimal N and D vs compute budget. The straight lines confirm power law behavior. Note how both scale at roughly C^0.5.*

### The Bitter Truth: Validation Loss Curves

When you plot validation loss (BPB - bits per byte) vs model size at a fixed compute budget, you get a U-shaped curve. Too small = underfitting. Too big = not enough training tokens.

<p align="center">
  <img src="scripts/graphs/validation_bpb_curve.png" width="100%"/>
</p>

*Figure: Validation BPB vs model parameters for four compute budgets. Stars mark the optimal models from curve fitting. Notice how the optimal point shifts right (bigger models) as compute increases.*

Key observations:

1. **There's a sweet spot**: For 1e18 FLOPs, optimal is ~154M params with 7.05 tokens/param. For 6e18 FLOPs, it's ~341M params with 8.61 tokens/param.

2. **Bigger budgets favor bigger models**: As C increases, the optimal N grows, but not as fast as C itself (remember, N ∝ C^0.456).

3. **More tokens per param at scale**: The optimal tokens/param ratio increases slightly with compute (7.05 → 8.61). But it's still way below Chinchilla's 20 tokens/param.

### Wait, What About Chinchilla?

Chinchilla (Hoffmann et al., 2022) famously said: "for optimal compute efficiency, you should train with 20 tokens per parameter." Our fitted curves suggest much lower ratios (7-9 tokens/param). 

What's going on? A few possibilities:

1. **Different compute regime**: Chinchilla studied 400M-70B param models. We're at 76M-522M. Scaling laws can have different exponents at different scales.

2. **Optimizer differences**: We use DistMuon (momentum + Newton-Schulz orthogonalization). Chinchilla used AdamW. Better optimizers can extract more from fewer tokens.

3. **Data quality**: FineWeb-Edu is higher quality than raw Common Crawl. Quality tokens count for more.

4. **Curve fitting vs actual runs**: Our optima come from spline fits through empirical points. There's uncertainty in the interpolation.

The honest answer? Scaling laws are empirical, not theoretical. They depend on architecture, optimizer, data, and the range you're measuring. Our laws apply to *our* setup. If you change something, re-run the experiments!

### Practical Takeaways

If you have a fixed compute budget and want to train optimally:

1. **Don't train too long**: The "train forever on a tiny model" approach is suboptimal. Scale up the model size.

2. **Don't go too wide**: A huge model trained for 100 steps won't work either. Balance N and D.

3. **Use the fitted curve**: Our power law fits let you extrapolate. Want to train with 1e19 FLOPs? Optimal is ~750M params (extrapolating N ∝ C^0.456).

4. **Empiricism over theory**: Scaling laws are discovered, not derived. When in doubt, run the experiment.

### Running Your Own Scaling Law Study

```bash
# Sweep depths and FLOP budgets
make run-scaling-law

# Plot the results
uv run python scripts/plot_isoflop_curve.py

# Get beautiful curves and optimal points
```

The script automatically:
- Finds all log files matching `scaling_laws_N*_F*.log`
- Extracts validation BPB, params, tokens for each run
- Fits smooth curves (cubic splines) to find optima
- Plots isoflop curves, validation curves, and optimal scaling

You'll see console output with fitted exponents and optimal parameters for each budget. Use these to plan your next training run.

## Architecture (GPT-2 124M)

```
- 12 layers, 12 heads, 768 dims
- 1024 context length
- 50257 vocab (GPT-2 BPE)
- ~124M parameters
```

Training setup:
```
- DistMuonAdamW optimizer (from Andrej's nanoGPT: hybrid Muon+AdamW with ZeRO-2)
- Learning rate: 6e-4 → 6e-5 (cosine decay)
- Batch size: 524,288 tokens (2^19)
- Weight decay: 0.1
- Warmup: 715 steps
- Total: 19,531 steps (10B tokens)
```

**Optimizer note**: Uses Muon (with gradient orthogonalization) for weight matrices, AdamW for embeddings. The optimizer state is sharded across GPUs to save memory. See [docs/README_DISTMUON.md](docs/README_DISTMUON.md) for details.

## Datasets: What to train on?

**FineWeb-Edu** (~10B tokens) - Start here! 🌟

High-quality educational content from the web (think Wikipedia, educational blogs, tutorials). Your model learns how language works and picks up world knowledge. Train for one epoch and you're done.

```bash
cd data/fineweb_edu && uv run python prepare_parquet.py --config sample-10BT
make ddp-train NGPUS=8 MODE=pretraining
```

**TaskMixture** (~460M tokens) - For chatbots 💬

Mix of SmolTalk (conversations), MMLU (reasoning), and GSM8K (math). Teaches your base model to be an assistant that follows instructions.

```bash
cd data/task_mixture && uv run python prepare.py
make ddp-train NGPUS=8 MODE=mid-training CHECKPOINT=/path/to/base_model.pt
```

**OpenWebText** (legacy) - For GPT-2 purists 📚

The original GPT-2 training data replica. Still works, but FineWeb-Edu is higher quality.

## Training modes

**pretraining** - Start from random weights, learn language
```bash
make ddp-train NGPUS=8 MODE=pretraining
```
Output: A model that can complete any text but won't follow instructions

**mid-training** - Turn your base model into a chatbot
```bash
make ddp-train NGPUS=8 MODE=mid-training CHECKPOINT=logs/pretraining/step_19531.pt
```
Output: An assistant that responds to "Explain quantum mechanics"

**all** - Do both in one command
```bash
make ddp-train NGPUS=8 MODE=all
```
For when you want to go from zero to chatbot in one run.

## Evaluation: How good is it?

We include 35+ benchmarks so you can see real numbers:

```bash
# Base models: MMLU, HellaSwag, PIQA, WinoGrande, ARC, etc.
make ddp-train NGPUS=8 MODE=pretraining CORE_EVALS=true

# Chat models: MT-Bench, AlpacaEval, IFEval
make ddp-train NGPUS=8 MODE=mid-training CHATCORE_EVALS=true CHECKPOINT=...
```

Evals run periodically during training on rank 0 (saves GPU time). Scores are rescaled: 0% = random guessing, 100% = perfect.

📖 **Full details**: [resources/eval_bundle/EVAL_GAUNTLET.md](resources/eval_bundle/EVAL_GAUNTLET.md)

## Chat with your model

Command line:
```bash
uv run python scripts/chat.py --checkpoint logs/pretraining/step_19531.pt
```

Web UI (much nicer):
```bash
make chat-server  # Opens at http://localhost:8003
```

For base models you get raw text continuation. For mid-trained models you get proper chat formatting with system/user/assistant roles.

## Performance notes

**Speed optimizations**:
- Mixed precision (bfloat16)
- Flash Attention
- Memory-mapped data loading: Zero-copy I/O

**Memory**: DistMuon's ZeRO-2 sharding saves optimizer memory by `1/world_size`
- Optimizer state sharded across GPUs
- Same-shape parameter batching for faster Muon updates
- No DDP wrapper needed

**Scaling**: Works with 1 to 8 GPUs out of the box
- Gradient accumulation auto-adjusts to maintain batch size
- Same final model regardless of GPU count

## Troubleshooting

**Out of memory?** Reduce batch size: `BATCH_SIZE=32` (default is 64)

**Slow data loading?** Your filesystem might not support mmap (common on NFS). Copy data to local SSD.

**NaN loss?** Learning rate too high (try 3e-4) or corrupted data.

**GPUs stuck with zombie processes?** `make kill-gpu` to clear them.

**NCCL errors?** Check `make gpu-status` and verify NCCL is compiled: `python -c "import torch; print(torch.cuda.nccl.version())"`

## Finetuning on your own data

1. Format as text files (one doc per line, or use delimiters)
2. Copy `data/fineweb_edu/prepare_parquet.py` and modify for your data
3. Save as Parquet shards (see prepare_parquet.py for format details)
4. Update config.py with your data directory path

For chat format, use special tokens `<|im_start|>` and `<|im_end|>` (see TaskMixture for examples).

## Makefile helpers

Run `make help` to see all available commands.

Quick reference:
```bash
make environment      # Setup
make ddp-train        # Train
make gpu-status       # Check GPUs
make format          # Format code
```

## Documentation

We wrote guides explaining everything in detail:

| Guide | What's in it |
|-------|--------------|
| [README_OPTIMIZATION.md](docs/README_OPTIMIZATION.md) | Momentum, Nesterov, weight decay, Muon - from first principles |
| [README_DISTMUON.md](docs/README_DISTMUON.md) | How the distributed optimizer works (ZeRO-2, sharding, no DDP) |
| [README_DEPTH_PARAMETERIZATION.md](docs/README_DEPTH_PARAMETERIZATION.md) | The DEPTH knob, auto-scaling, scaling laws |
| [README_MUON.md](docs/README_MUON.md) | Muon optimizer internals (Newton-Schulz orthogonalization) |
| [README_FLOPS_AND_ITERATIONS.md](docs/README_FLOPS_AND_ITERATIONS.md) | FLOPs calculation, compute budgets |
| [README_ROPE.md](docs/README_ROPE.md) | Rotary position embeddings |
| [README_STABILITY.md](docs/README_STABILITY.md) | Gradient clipping, QK-Layernorm, Z-loss |
| [README_MULTIPLEX.md](docs/README_MULTIPLEX.md) | Multi-dataset training (how TaskMixture works) |
| [README_CHATCORE_EVALUATOR.md](docs/README_CHATCORE_EVALUATOR.md) | Chat model evaluation (MT-Bench, AlpacaEval, IFEval) |
| [README-PREFETCHING-OPTIMIZATION.md](docs/README-PREFETCHING-OPTIMIZATION.md) | BOS-aligned dataloader async prefetching (eliminating MFU jitter) |

**New to training LLMs?** Start with README_OPTIMIZATION.md to understand the basics, then README_DISTMUON.md for distributed training.

## Contributing

The codebase is designed to be simple and hackable. If you want to:
- Try a new optimizer
- Add a new dataset
- Implement a new architecture feature
- Improve the evaluation suite

...the code should be straightforward to modify. Pull requests welcome!

## Credits and references

**Giants whose shoulders we're standing on:**

- **Andrej Karpathy**: [nanoGPT](https://github.com/karpathy/nanoGPT) (the foundation), [nanochat](https://github.com/karpathy/nanochat) (depth parameterization, scaling laws). Seriously, go star those repos.
- **Vaswani et al.**: [Attention is All You Need](https://arxiv.org/abs/1706.03762) (2017) - The transformer paper
- **Radford et al.**: [GPT-2 paper](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) (2019)

**Datasets:**
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) - HuggingFace's excellent filtered Common Crawl
- [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) - GPT-2 training data replica
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) - Conversational data
- [MMLU](https://huggingface.co/datasets/cais/mmlu) - Reasoning benchmark
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) - Math problems

**Evaluation:**
- [Mosaic Eval Gauntlet](https://www.mosaicml.com/blog/llm-evaluation-for-icl) - Comprehensive benchmark suite

## License

MIT

---

*Questions? Issues? Want to share what you built? Open an issue or PR!*
