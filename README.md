# NanoGPT

The simplest, fastest repository for training/finetuning GPT-2 (124M). A rewrite of the original [nanoGPT](https://github.com/karpathy/nanoGPT) with modern datasets, distributed training with memory-efficient optimizers, and comprehensive evaluation.

**What's new:** DistMuon hybrid optimizer (Muon + AdamW with ZeRO-2 sharding), depth-based parameterization for scaling laws, isoflop curve analysis, and 35+ evaluation benchmarks.

## Install

```bash
make environment  # installs uv and creates venv
```

Dependencies: `pytorch`, `numpy`, `transformers` (for tokenization), `datasets` (for data loading), `wandb` (for logging), `tiktoken`.

## Quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a GPT on FineWeb-Edu. This will give you a model that understands language and can complete text.

```bash
# Step 1: Prepare data (~30-60 min download and tokenization)
cd data/fineweb_edu && uv run python prepare.py

# Step 2: Train (single GPU - slower but works on any setup)
make ddp-train NGPUS=1 MODE=pretraining

# Or train with multiple GPUs (much faster, e.g. 8 GPUs)
make ddp-train NGPUS=8 MODE=pretraining
```

**Time estimates:**
- Single GPU (e.g., A100): ~30 days for 10B tokens
- 8 GPUs: ~4 days for 10B tokens (~19K steps)
- The model checkpoints save to `logs/` every few thousand steps, so you can stop early and still have a working model.

**Chat with your model:**

```bash
uv run python scripts/chat.py --checkpoint logs/pretraining/step_19531.pt
```

Or start the web UI (more user-friendly):

```bash
make chat-server  # Opens at http://localhost:8003
```

## Depth Parameterization (Scaling Laws)

NanoGPT supports **depth-based parameterization** inspired by [nanochat](https://github.com/karpathy/nanochat), which simplifies model scaling using a single knob:

```bash
# Train models of different sizes with one parameter
make ddp-train DEPTH=6 TARGET_FLOPS=1e18   # ~30M params
make ddp-train DEPTH=12 TARGET_FLOPS=1e18  # ~150M params
make ddp-train DEPTH=20 TARGET_FLOPS=1e18  # ~560M params
```

**Benefits:**
- 🎯 **Single knob:** `DEPTH` controls both model width and depth
- 📊 **Auto-scaling:** Learning rate and weight decay scale automatically
- 🔬 **Easy sweeps:** Run scaling law experiments with simple loops

**Run full scaling law experiments:**
```bash
make run-scaling-law  # Sweeps depths 6-14 and FLOP budgets 1e18-6e18
```

**Visualize results:**
```bash
# Plot Chinchilla-style ISOFlop curves (loss vs model size for fixed compute)
uv run python scripts/plot_isoflop_curve.py

# Plot training time vs model size
uv run python scripts/plot_training_time.py
```

The ISOFlop curve analysis helps you find the optimal model size for your compute budget - just like in the Chinchilla paper. The scripts automatically parse your logs and generate publication-ready plots.

📖 **See [docs/README_DEPTH_PARAMETERIZATION.md](docs/README_DEPTH_PARAMETERIZATION.md) for full documentation**

## Reproducing GPT-2 (124M)

This implementation reproduces GPT-2 (124M parameters) with the following architecture:

```
- 12 layer transformer
- 12 attention heads  
- 768 embedding dimension
- 1024 context length
- 50257 vocabulary size (GPT-2 BPE)
```

The model is trained with:

```
- DistMuonAdamW optimizer (hybrid Muon + AdamW with ZeRO-2 sharding)
- learning rate 6e-4, cosine decay to 6e-5
- 715 warmup steps
- batch size 524,288 tokens (2^19)
- weight decay 0.1
- gradient clipping 1.0
- bfloat16 mixed precision
```

**About the optimizer:** We use a hybrid approach - Muon (with gradient orthogonalization) for 2D weight matrices (attention, MLP), and AdamW for embeddings. The optimizer state is sharded across GPUs (ZeRO-2 style) to reduce memory usage by `1/world_size`. See [docs/README_DISTMUON.md](docs/README_DISTMUON.md) for details.

Training runs to 19,531 steps (10B tokens), which is roughly 1 epoch over FineWeb-Edu. With 8 A100 GPUs you can expect ~4 days of training.

## Datasets

The repository supports three datasets, each serving a different purpose:

**FineWeb-Edu** (~10B tokens) - **Recommended for pretraining.** High-quality educational web content from Common Crawl, filtered and deduplicated by HuggingFace. Think Wikipedia-quality articles, educational blogs, tutorials. This teaches your model general language understanding and world knowledge. One epoch is enough - after that you start overfitting.

**TaskMixture** (~568K examples) - **For mid-training/instruction tuning.** Combines SmolTalk (conversational), MMLU (reasoning), and GSM8K (math). Formatted with chat special tokens (`<|im_start|>`, `<|im_end|>`). This teaches your pretrained model how to follow instructions and have conversations.

**OpenWebText** (legacy) - Earlier GPT-2 reproduction dataset. Still supported but FineWeb-Edu is higher quality. Use this if you want to match GPT-2 original training more closely.

### TaskMixture Data Breakdown

**Training Split** (567,315 examples | 460.3M tokens):

| Dataset | Examples | Percentage | Description |
|---------|----------|------------|-------------|
| SmolTalk | 460,000 | 81.1% | General conversations |
| MMLU auxiliary_train | 99,842 | 17.6% | Multiple choice (ARC, MC_TEST, OBQA, RACE) |
| GSM8K | 7,473 | 1.3% | Grade school math problems |
| **TOTAL** | **567,315** | **100%** | **460.3M tokens** |

**Validation Split** (7,850 examples | 5.0M tokens):

| Dataset | Examples | Percentage | Description |
|---------|----------|------------|-------------|
| SmolTalk | 5,000 | 63.7% | Sampled conversations |
| MMLU | 1,531 | 19.5% | Validation subset ("all" config) |
| GSM8K | 1,319 | 16.8% | Test split for validation |
| **TOTAL** | **7,850** | **100%** | **5.0M tokens** |


All datasets are tokenized with GPT-2's BPE tokenizer and stored as binary `.bin` files (memory-mapped uint16 arrays) for fast loading.

## Baselines

**What to expect:**
- **FineWeb-Edu pretraining:** Validation loss around 2.8-3.0 after 10B tokens (1 epoch). This is a strong baseline - FineWeb-Edu is high-quality educational content, so you're already learning from curated data.
- **Mid-training on TaskMixture:** Further improves instruction following and reasoning. Validation loss will initially increase (you're changing domains), then improve as the model adapts to conversational format.
- **Perplexity:** ~16-20 on FineWeb-Edu validation set (exp(2.8-3.0))

These numbers are for the 124M parameter model. Smaller models will have higher loss, larger models lower loss (see scaling laws).

## Data prep

Each dataset has its own preparation script. Run these from inside the dataset directories:

```bash
# FineWeb-Edu (primary)
cd data/fineweb_edu
uv run python prepare.py

# TaskMixture (for mid-training)
cd data/task_mixture  
uv run python prepare.py

# OpenWebText (legacy)
cd data/openwebtext
uv run python prepare.py
```

Data prep will download the dataset from HuggingFace, tokenize it with GPT-2's tokenizer (plus special tokens where applicable), and save train/val splits as `.bin` files.

Note: If you're on a network filesystem that doesn't support mmap (like NFS), the dataloader will fall back to regular file reads. For best performance, copy data to local SSD.

## Training modes

The trainer supports three modes, matching the typical LLM training pipeline:

**pretraining** - Train from scratch on FineWeb-Edu (start here!)
```bash
make ddp-train NGPUS=8 MODE=pretraining
```
This is language modeling on web text. Your model learns grammar, facts, reasoning patterns. Output: a base model that can complete any text but doesn't follow instructions.

**mid-training** - Continue from a checkpoint on TaskMixture (instruction tuning)
```bash
make ddp-train NGPUS=8 MODE=mid-training CHECKPOINT=/path/to/checkpoint.pt
```
This teaches your base model to be a chatbot. It learns the conversational format and how to answer questions. Output: an assistant that responds to prompts like "Explain quantum mechanics."

**all** - Run pretraining followed by mid-training (full pipeline)
```bash
make ddp-train NGPUS=8 MODE=all
```
Do everything in one command. Useful for end-to-end experiments or scaling law sweeps.

**Note:** Adjust `NGPUS` to match your GPU count. The system automatically calculates gradient accumulation steps to maintain the target batch size of 524,288 tokens, so you get the same effective batch size whether you use 1 or 8 GPUs (training will just be faster with more GPUs).

## Sampling / inference

Use the chat script for interactive generation:

```bash
uv run python scripts/chat.py --checkpoint /path/to/checkpoint.pt
```

Or launch the web UI server:

```bash
make chat-server  # starts server on http://localhost:8003
```

For mid-trained models, the chat interface supports special tokens for system/user/assistant formatting. For pretrained models, you get raw text continuation.

The CLI script uses a simple sampling scheme with temperature 0.9 and top-k 200. You can modify these in the script.

## Benchmarking

The codebase includes comprehensive evaluation via the **Mosaic Eval Gauntlet** - 35+ benchmarks across reading comprehension, commonsense reasoning, world knowledge, math, language understanding, and safety. These run periodically during training so you can see how your model improves.

**For pretraining (base models):**

Run with core evaluations enabled:
```bash
make ddp-train NGPUS=8 MODE=pretraining CORE_EVALS=true
```

This includes MMLU, HellaSwag, PIQA, WinoGrande, ARC, etc. Good for measuring general capabilities.

**For mid-training (instruction-tuned models):**

```bash
make ddp-train NGPUS=8 MODE=mid-training CHATCORE_EVALS=true CHECKPOINT=/path/to/checkpoint.pt
```

This includes MT-Bench, AlpacaEval, IFEval - benchmarks designed for conversational models.

**Skip evaluations for faster training:**

```bash
make ddp-train NGPUS=8 MODE=pretraining VAL_EVALS=false
```

Useful when you just want to train fast and don't need continuous evaluation.

**Note:** Evaluations run only on rank 0 (to save GPU time) and report scores rescaled above random baseline. This means 0% = random guessing, 100% = perfect score. See `resources/eval_bundle/EVAL_GAUNTLET.md` for full details.

## Efficiency notes

**Multi-GPU training:**
- DistMuon optimizer with ZeRO-2 style sharding (no DDP wrapper needed)
- Optimizer state sharded across GPUs - reduces memory by `1/world_size`
- Same-shape parameter batching for ~10x faster Muon updates via fused kernels

**Speed optimizations:**
- Mixed precision (bfloat16) for ~2x speedup and 50% memory reduction
- Flash Attention where available (requires PyTorch 2.0+)
- Memory-mapped data loading for zero-copy I/O
- Gradient checkpointing not used (124M is small enough to fit)

With 8xA100 (40GB) you should see ~50K tokens/sec throughput with the default batch size of 64 per GPU.

**Memory tip:** The optimizer state sharding means you can train larger models or use bigger batch sizes than with standard DDP. If you still run out of memory, reduce the per-GPU batch size - the system will automatically increase gradient accumulation steps to maintain the target total batch size.

## Finetuning

For finetuning on your own data:

1. Format your data as text files (one document per line, or use delimiters)
2. Create a preparation script similar to `data/fineweb_edu/prepare.py`
3. Tokenize and save as train.bin / val.bin
4. Update the data path in your config or pass `--data_dir`
5. Use mid-training mode with a pretrained checkpoint

The mid-training setup (TaskMixture) provides a good template for instruction-following datasets. Use special tokens `<|im_start|>` and `<|im_end|>` to mark role boundaries.

## Makefile helpers

Run `make help` to see all available options.

## Troubleshooting

**Out of memory**: Reduce batch size per GPU (e.g., `BATCH_SIZE=32` instead of 64). The gradient accumulation will be auto-adjusted to maintain the target total batch size. Remember: the DistMuon optimizer already saves memory by sharding optimizer state.

**Slow data loading**: Network filesystems (like NFS) often don't support mmap efficiently. Copy data to local SSD for ~10x faster loading.

**NCCL errors**: The distributed optimizer needs working NCCL for reduce-scatter and all-gather. Check that:
  - All GPUs are visible: `make gpu-status`
  - NCCL is compiled: `python -c "import torch; print(torch.cuda.nccl.version())"`
  - No firewall blocking inter-GPU communication

**GPUs busy with zombie processes**: Use `make kill-gpu` to clear all GPU processes. Useful when training crashes but processes linger.

**NaN loss**: Usually indicates:
  - Learning rate too high (try 3e-4 instead of 6e-4)
  - Corrupted data (check your dataset preprocessing)
  - Mixed precision issues (rare with bfloat16, but check for inf gradients)

**Loss not decreasing**: Check that you're using the right dataset paths and that tokenization completed successfully. Look for `train.bin` and `val.bin` files in your data directory.

## Future work

Some ideas for future improvements:

- [ ] **Multi-node training**: Currently supports single-node multi-GPU. Extending DistMuon to multi-node would enable training larger models.
- [ ] **Gradient checkpointing**: For scaling beyond 124M params without OOM errors.
- [ ] **Alternative tokenizers**: Support Llama, GPT-NeoX, or custom vocabularies.
- [ ] **Quantization-aware training**: QAT for efficient deployment (INT8/INT4).
- [ ] **KV cache optimizations**: PagedAttention or continuous batching for serving.
- [ ] **MoE (Mixture of Experts)**: Scale to billions of params with sparse routing.
- [ ] **FlashAttention-3**: When it becomes available for even faster attention.
- [ ] **Longer context**: Extend beyond 1024 tokens with efficient attention variants.

Contributions welcome! The codebase is designed to be simple and hackable.

## References

This code is based on:

- [nanoGPT](https://github.com/karpathy/nanoGPT) by Andrej Karpathy - the original simple GPT implementation
- [Attention is All You Need](https://arxiv.org/abs/1706.03762) - Vaswani et al. 2017, the transformer paper
- [Language Models are Unsupervised Multitask Learners](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) - Radford et al. 2019, GPT-2 paper

Datasets:
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) - HuggingFace filtered Common Crawl
- [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) - GPT-2 training set reproduction
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) - Conversational data
- [MMLU](https://huggingface.co/datasets/cais/mmlu) - Measuring Massive Multitask Language Understanding
- [GSM8K](https://huggingface.co/datasets/openai/gsm8k) - Grade School Math 8K

Evaluation:
- [Mosaic Eval Gauntlet](https://www.mosaicml.com/blog/llm-evaluation-for-icl) - Comprehensive benchmark suite

## Documentation

Comprehensive guides are available in the `docs/` folder:

| Guide | Description |
|-------|-------------|
| [README_OPTIMIZATION.md](docs/README_OPTIMIZATION.md) | **Complete optimization guide**: Understanding momentum, Nesterov, weight decay, and Muon optimizer from first principles. Includes practical tips, hyperparameters, and debugging strategies. |
| [README_MUON.md](docs/README_MUON.md) | **Muon optimizer**: Second-order-ish optimizer using Newton-Schulz orthogonalization. Explains why we use separate optimizers for matrix vs. embedding parameters. |
| [README_DISTMUON.md](docs/README_DISTMUON.md) | **DistMuon distributed optimizer**: How the hybrid Muon+AdamW optimizer works with ZeRO-2 style sharding, parameter batching for speed, and distributed training without DDP. |
| [README_DEPTH_PARAMETERIZATION.md](docs/README_DEPTH_PARAMETERIZATION.md) | **Depth-based scaling**: How the single `DEPTH` parameter controls model architecture, automatic LR/WD scaling, and running scaling law experiments. |
| [README_FLOPS_AND_ITERATIONS.md](docs/README_FLOPS_AND_ITERATIONS.md) | **FLOPs calculation**: How we calculate training FLOPs, convert between tokens/steps/FLOPs, and use `TARGET_FLOPS` for reproducible experiments. |
| [README_ROPE.md](docs/README_ROPE.md) | **Rotary Position Embeddings (RoPE)**: How RoPE works, why it's better than absolute positional encodings, implementation details. |
| [README_STABILITY.md](docs/README_STABILITY.md) | **Training stability**: Gradient clipping, QK-Layernorm, Z-loss regularization, and other techniques for stable training at scale. |
| [README_MULTIPLEX.md](docs/README_MULTIPLEX.md) | **Multiplex dataloader**: How we combine multiple datasets (SmolTalk, MMLU, GSM8K) with weighted sampling for mid-training. |
| [README_CHATCORE_EVALUATOR.md](docs/README_CHATCORE_EVALUATOR.md) | **ChatCORE evaluation**: Specialized benchmarks for instruction-tuned models (MT-Bench, AlpacaEval, IFEval, FLASK). |
| [README_VLM_GUIDE.md](docs/README_VLM_GUIDE.md) | **Vision-Language Models**: Future extension guide for adding vision capabilities to the GPT model. |

**Getting started?** Read [README_OPTIMIZATION.md](docs/README_OPTIMIZATION.md) to understand the training hyperparameters, then [README_DISTMUON.md](docs/README_DISTMUON.md) for the distributed optimizer, and [README_DEPTH_PARAMETERIZATION.md](docs/README_DEPTH_PARAMETERIZATION.md) for running scaling experiments.

## License

MIT
