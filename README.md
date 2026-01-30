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
cd data/fineweb_edu && uv run python prepare.py && cd ../..

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
make run-scaling-law  # Sweeps depths 6-14, FLOP budgets 1e18-6e18
uv run python scripts/plot_isoflop_curve.py  # Plot results
```

You'll get beautiful Chinchilla-style plots showing optimal model size for your compute budget.

📖 **More details**: [docs/README_DEPTH_PARAMETERIZATION.md](docs/README_DEPTH_PARAMETERIZATION.md)

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
cd data/fineweb_edu && uv run python prepare.py
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
2. Copy `data/fineweb_edu/prepare.py` and modify for your data
3. Tokenize and save as `train.bin` / `val.bin`
4. Run mid-training mode with a pretrained checkpoint

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
