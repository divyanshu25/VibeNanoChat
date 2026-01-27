# NanoGPT

The simplest, fastest repository for training/finetuning GPT-2 (124M). A rewrite of the original [nanoGPT](https://github.com/karpathy/nanoGPT) with modern datasets, distributed training support, and comprehensive evaluation.

## Install

```bash
make environment  # installs uv and creates venv
```

Dependencies: `pytorch`, `numpy`, `transformers` (for tokenization), `datasets` (for data loading), `wandb` (for logging), `tiktoken`.

## Quick start

If you are not a deep learning professional and you just want to feel the magic and get your feet wet, the fastest way to get started is to train a GPT on FineWeb-Edu:

```bash
# prepare data (~30-60 min download and tokenization)
cd data/fineweb_edu && uv run python prepare.py

# train (single GPU)
make ddp-train NGPUS=1 MODE=pretraining

# train (multi-GPU, e.g. 8 GPUs)
make ddp-train NGPUS=8 MODE=pretraining
```

Training with 8 GPUs takes approximately 4 days to reach 10B tokens (~19K steps). The model will be saved periodically to the `logs/` directory.

To chat with your trained model:

```bash
uv run python scripts/chat.py --checkpoint /path/to/checkpoint.pt
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

📖 **See [docs/DEPTH_PARAMETERIZATION.md](docs/DEPTH_PARAMETERIZATION.md) for full documentation**

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
- AdamW optimizer
- learning rate 6e-4, cosine decay to 6e-5
- 715 warmup steps
- batch size 524,288 tokens (2^19)
- weight decay 0.1
- gradient clipping 1.0
- bfloat16 mixed precision
```

Training runs to 19,531 steps (10B tokens), which is roughly 1 epoch over FineWeb-Edu. With 8 A100 GPUs you can expect ~4 days of training.

## Datasets

The repository supports three datasets:

**FineWeb-Edu** (~10B tokens) - Recommended for pretraining. High quality educational web content from Common Crawl, filtered and deduplicated by HuggingFace. This is the primary dataset used for pretraining.

**TaskMixture** (~568K examples) - For mid-training/instruction tuning. Combines SmolTalk (conversational), MMLU (reasoning), and GSM8K (math). Formatted with chat special tokens.

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

FineWeb-Edu is a strong dataset. You should expect validation loss around 2.8-3.0 after full pretraining. Mid-training on TaskMixture further improves instruction following and reasoning capabilities.

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

The trainer supports three modes:

**pretraining** - Train from scratch on FineWeb-Edu
```bash
make ddp-train NGPUS=8 MODE=pretraining
```

**mid-training** - Continue from a checkpoint on TaskMixture
```bash
make ddp-train NGPUS=8 MODE=mid-training CHECKPOINT=/path/to/checkpoint.pt
```

**all** - Run pretraining followed by mid-training
```bash
make ddp-train NGPUS=8 MODE=all
```

Adjust `NGPUS` to match your GPU count. The system automatically calculates gradient accumulation steps to maintain the target batch size of 524,288 tokens.

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

The codebase includes comprehensive evaluation via the Mosaic Eval Gauntlet - 35+ benchmarks across reading comprehension, commonsense reasoning, world knowledge, math, language understanding, and safety.

Run with core evaluations enabled:
```bash
make ddp-train NGPUS=8 MODE=pretraining CORE_EVALS=true
```

Or disable all evaluations for faster training:
```bash
make ddp-train NGPUS=8 MODE=pretraining VAL_EVALS=false
```

For mid-training with chat-focused evaluations:
```bash
make ddp-train NGPUS=8 MODE=mid-training CHATCORE_EVALS=true CHECKPOINT=/path/to/checkpoint.pt
```

Evaluations run only on rank 0 and report scores rescaled above random baseline. See `resources/eval_bundle/EVAL_GAUNTLET.md` for details.

## Efficiency notes

- Uses PyTorch's DDP for multi-GPU training
- Mixed precision (bfloat16) for ~2x speedup and 50% memory reduction
- Flash Attention where available (requires PyTorch 2.0+)
- Memory-mapped data loading for zero-copy I/O
- Gradient checkpointing not used (124M is small enough to fit)

With 8xA100 (40GB) you should see ~50K tokens/sec throughput with the default batch size of 64 per GPU.

If you run out of memory, reduce the per-GPU batch size in the config. The system will automatically increase gradient accumulation steps to maintain the target total batch size.

## Finetuning

For finetuning on your own data:

1. Format your data as text files (one document per line, or use delimiters)
2. Create a preparation script similar to `data/fineweb_edu/prepare.py`
3. Tokenize and save as train.bin / val.bin
4. Update the data path in your config or pass `--data_dir`
5. Use mid-training mode with a pretrained checkpoint

The mid-training setup (TaskMixture) provides a good template for instruction-following datasets. Use special tokens `<|im_start|>` and `<|im_end|>` to mark role boundaries.

## Makefile helpers

```bash
# Setup
make environment                    # install uv and setup venv
make jupyter-kernel                 # register as jupyter kernel

# Training
make ddp-train NGPUS=8 MODE=pretraining                           # basic training
make ddp-train NGPUS=8 MODE=mid-training CHECKPOINT=/path/to/pt  # mid-training
make ddp-train NGPUS=8 MODE=pretraining CORE_EVALS=true          # with evals

# GPU Management
make gpu-status                     # check nvidia-smi
make kill-gpu                       # kill all processes on GPUs
make gpu-hot GPUS=0,1,2            # keep specific GPUs active
make gpu-hot GPUS=0,1,2 DELAY=2    # start heating in 2 hours

# Code Quality
make format                         # format with black and isort
make lint                           # lint with ruff
make check                          # format + lint
```

## Troubleshooting

**Out of memory**: Reduce batch size per GPU. The gradient accumulation will be auto-adjusted.

**Slow data loading**: Network filesystems often don't support mmap efficiently. Copy data to local disk.

**NCCL errors**: Ensure all GPUs are visible (`make gpu-status`) and NCCL is compiled correctly (`python -c "import torch; print(torch.cuda.nccl.version())"`).

**GPUs busy with zombie processes**: Use `make kill-gpu` to clear all GPU processes.

**NaN loss**: Usually indicates learning rate too high or corrupted data. Try reducing LR or checking your dataset.

## Todos

Some future ideas:

- [ ] Multi-node training support (currently single-node multi-GPU only)
- [ ] Gradient checkpointing option for training larger models
- [ ] Support for other tokenizers (e.g. Llama, GPT-NeoX)
- [ ] Quantization-aware training
- [ ] KV cache optimization for faster inference
- [ ] Direct integration with more evaluation frameworks

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

## License

MIT
