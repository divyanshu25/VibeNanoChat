# VibeNanoChat

> **Standing on the shoulders of giants**: This repo is built on top of Andrej Karpathy's excellent [nanoGPT](https://github.com/karpathy/nanoGPT) and [nanochat](https://github.com/karpathy/nanochat). Most of the good ideas here (depth parameterization, training setup, elegant code structure) come directly from Andrej's work. What we've added: distributed training with ZeRO-style sharding, modern datasets, and evaluation tools. All credit for the foundation goes to @karpathy! 🙏

Train your own GPT from scratch in ~4 days on 8 GPUs. Simple, hackable code. No enterprise complexity, just pure PyTorch.

## Why this repo?

**For learners**: You want to understand how GPT works by training one yourself. The code is clean, well-commented, and each component does one thing well.

**For researchers**: You want to run scaling law experiments or try new ideas without wading through production codebases. We have depth parameterization (thanks Andrej!), automatic scaling, and isoflop curves ready to go.

**For tinkerers**: You want something that *just works* but is still easy to modify. Change the architecture, try new optimizers, plug in your own data - it's all straightforward.

## What you get

- 🚀 Train GPT-2 (124M) in ~4 days on 8xA100 (or 30 days on 1 GPU if you're patient)
- 🎯 One-knob scaling: `DEPTH=12` controls everything, LR/WD auto-scale
- 📊 Built-in evaluation: 35+ benchmarks (MMLU, HellaSwag, etc.)
- 💬 Chat with your model: Web UI included
- 🔬 Scaling law tools: Run experiments and plot Chinchilla-style curves
- 🌐 Modern datasets: FineWeb-Edu (quality!), TaskMixture for instruction tuning

## Quick start: Train your first GPT

```bash
# Install (takes ~2 min)
make environment

# Get some data (~30-60 min download + tokenization)
cd data/fineweb_edu && uv run python prepare.py && cd ../..

# Train! (4 days on 8 GPUs, or 30 days on 1 GPU)
make ddp-train NGPUS=8 MODE=pretraining

# Chat with it
uv run python scripts/chat.py --checkpoint logs/pretraining/step_19531.pt
```

That's it. You now have a GPT that understands language.

**Pro tip**: Models checkpoint every few thousand steps, so you can stop early and still have something that works. A model trained for just 1 day is already pretty interesting!

## What's actually new here?

Let's be honest about what we added vs what we borrowed:

**From nanoGPT** (Andrej Karpathy):
- The entire GPT-2 architecture and training loop
- Clean, minimal code philosophy
- Smart defaults that just work

**From nanochat** (also Andrej Karpathy):
- Depth parameterization (the `DEPTH` knob that scales everything)
- Scaling law experimental setup
- Auto-scaling of learning rate and weight decay

**What we added**:
- **DistMuon optimizer**: Hybrid Muon+AdamW with ZeRO-2 style sharding (train without DDP wrapper, save memory)
- **Modern datasets**: FineWeb-Edu (much better than OpenWebText), TaskMixture for chat
- **Evaluation suite**: 35+ benchmarks integrated into training
- **Makefile helpers**: Because typing long commands is annoying
- **Isoflop plotting**: Automatically generate Chinchilla-style curves from logs
- **Documentation**: Deep dives on optimization, distributed training, and scaling

So yeah, we're building on Andrej's foundation. The core insights are his. We just added some power tools.

## The Depth Knob 🎛️

This is Andrej's genius idea from nanochat: control model size with one parameter.

```bash
# Small model (~30M params)
make ddp-train DEPTH=6 TARGET_FLOPS=1e18

# Medium model (~150M params) - this is roughly GPT-2 size
make ddp-train DEPTH=12 TARGET_FLOPS=1e18

# Large model (~560M params)
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
- DistMuonAdamW optimizer (our contribution: hybrid with ZeRO-2)
- Learning rate: 6e-4 → 6e-5 (cosine decay)
- Batch size: 524,288 tokens (2^19)
- Weight decay: 0.1
- Warmup: 715 steps
- Total: 19,531 steps (10B tokens)
```

**Optimizer note**: We use Muon (with gradient orthogonalization) for weight matrices, AdamW for embeddings. The optimizer state is sharded across GPUs to save memory. This is our main technical contribution - see [docs/README_DISTMUON.md](docs/README_DISTMUON.md) for details.

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

**Expected numbers** for 124M pretraining:
- Validation loss: 2.8-3.0 (perplexity ~16-20)
- MMLU: Better than random, worse than GPT-3
- HellaSwag: Actually pretty decent!

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

**Speed**: ~50K tokens/sec on 8xA100 (40GB)
- Mixed precision (bfloat16): ~2x speedup
- Flash Attention: Even faster
- Memory-mapped data loading: Zero-copy I/O

**Memory**: DistMuon's ZeRO-2 sharding saves optimizer memory by `1/world_size`
- 8 GPUs: 8x less optimizer state per GPU
- Same-shape parameter batching: ~10x faster Muon updates
- No DDP wrapper needed

**Scaling**: Works with 1 to 8 GPUs out of the box
- Gradient accumulation auto-adjusts to maintain batch size
- Same final model regardless of GPU count (just faster with more GPUs)

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
