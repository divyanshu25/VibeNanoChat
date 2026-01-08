# 🚀 NanoGPT

> A production-ready implementation of GPT-2 (124M parameters) with comprehensive evaluation and multiple training modes. Train, evaluate, and chat with your own language model!

## ✨ What's Inside

- **🧠 GPT-2 Architecture** (124M parameters): 12 layers, 12 attention heads, 768 dimensions of pure transformer magic
- **⚡ Distributed Training**: Scale across multiple GPUs with PyTorch DDP
- **📚 Multiple Datasets**: 
  - FineWeb-Edu (~10B tokens of high-quality educational content) - Primary
  - TaskMixture (SmolTalk + MMLU + GSM8K) - For mid-training/alignment
  - OpenWebText (~9B tokens) - Legacy support
- **🎯 Three Training Modes**: Pretraining, mid-training, and full pipeline
- **📊 Comprehensive Evaluation**: 35+ benchmarks across 6 categories (Mosaic Eval Gauntlet)
- **🤖 Chat Interface**: Interactive chat with trained models
- **💡 Modern Training Stack**: Mixed precision (bfloat16), gradient clipping, cosine LR scheduling
- **📈 Experiment Tracking**: Built-in Weights & Biases integration
- **💾 Efficient Data Loading**: Memory-mapped binary files for lightning-fast I/O

## ⚡ Quick Start

```bash
# 1. Setup environment
make environment

# 2. Prepare dataset
cd data/fineweb_edu && uv run python prepare.py

# 3. Train with 8 GPUs
make ddp-train NGPUS=8 MODE=pretraining

# 4. Chat with your model
make chat CHECKPOINT=/path/to/checkpoint.pt
```

## 🎬 Getting Started

### 1. Clone & Setup

```bash
git clone https://github.com/yourusername/NanoGPT.git
cd NanoGPT
```

### 2. 🛠️ Install Dependencies

We use [UV](https://github.com/astral-sh/uv) because it's blazingly fast:

```bash
# 🎯 One command to rule them all
make environment

# Or step by step if you're old school:
make uv        # Install UV
make uvlock    # Lock dependencies
make venv      # Create virtual environment
```

### 3. 📊 Prepare Training Datasets

#### Option A: FineWeb-Edu (Recommended for Pretraining)

High-quality educational content (~10B tokens):

```bash
cd data/fineweb_edu
uv run python prepare.py
```

**What happens:**
- 📥 Downloads FineWeb-Edu dataset from HuggingFace
- 🔤 Tokenizes with GPT-2 BPE encoding + special tokens
- 💾 Saves to your configured data directory (configurable in script)
- ✅ Creates: `train.bin` and `val.bin` with billions of high-quality tokens

☕ This takes ~30-60 minutes depending on your connection!

#### Option B: TaskMixture (For Mid-Training/Alignment)

Specialized datasets for instruction following and reasoning:

```bash
cd data/task_mixture
uv run python prepare.py
```

**What happens:**
- 📥 Loads three datasets:
  - SmolTalk: Conversational data (~460K examples)
  - MMLU: Multiple choice reasoning (~100K examples)
  - GSM8K: Math word problems (~8K examples)
- 🔤 Formats with special tokens for chat/instruction following
- 💾 Saves to your configured data directory (configurable in script)
- ✅ Creates: `train.bin`, `val.bin`, and `metadata.json`

⏱️ Takes ~10-20 minutes to prepare

#### Option C: OpenWebText (Legacy)

Classic dataset for reproducibility (~9B tokens):

```bash
cd data/openwebtext
uv run python prepare.py
```

### 4. 🔥 Train the Model

NanoGPT supports three training modes:

#### Mode 1: Pretraining (Train from Scratch on FineWeb-Edu)

```bash
# Multi-GPU training (recommended)
make ddp-train NGPUS=8 MODE=pretraining

# Or with torchrun:
torchrun --standalone --nproc_per_node=8 src/gpt_2/ddp.py --mode pretraining

# Single GPU
python src/gpt_2/ddp.py --mode pretraining
```

#### Mode 2: Mid-Training (Instruction/Reasoning from Checkpoint)

Continue training from a pretrained checkpoint on TaskMixture:

```bash
make ddp-train NGPUS=8 MODE=mid-training CHECKPOINT=/path/to/checkpoint.pt

# Or with torchrun:
torchrun --standalone --nproc_per_node=8 src/gpt_2/ddp.py --mode mid-training --checkpoint /path/to/checkpoint.pt
```

#### Mode 3: Full Pipeline (Pretrain → Mid-Train)

Run both stages automatically:

```bash
make ddp-train NGPUS=8 MODE=all

# Or with torchrun:
torchrun --standalone --nproc_per_node=8 src/gpt_2/ddp.py --mode all
```

**⚙️ Training Configuration:**
- 📦 Batch size per GPU: 64
- 📏 Sequence length: 1024 tokens
- 🎯 Total batch size: 524,288 tokens/step (2^19, perfectly balanced)
- 🎓 Max learning rate: 6e-4 (pretraining), 3e-4 (mid-training)
- 🏃 Warmup steps: 715 (pretraining), 100 (mid-training)
- 💪 Optimizer: AdamW with weight decay 0.1
- ✂️ Gradient clipping: 1.0

## 🎛️ Configuration Deep Dive

### Model Config (GPT-2 124M)

```python
block_size: 1024      # Context window size
vocab_size: 50257     # GPT-2 vocabulary (BPE)
n_layer: 12           # Transformer blocks (the secret sauce)
n_head: 12            # Attention heads (parallel thoughts)
n_embed: 768          # Embedding dimension (the hidden state)
```

### Training Config

```python
max_learning_rate: 6e-4          # Peak LR (after warmup)
min_learning_rate: 6e-5          # Final LR (10% of max)
warmup_steps: 715                # Linear warmup phase
total_batch_size: 524288         # Tokens per optimization step
weight_decay: 0.10               # L2 regularization
gradient_clip_norm: 1.0          # Gradient explosion prevention
```

## 🎯 Evaluation Framework

NanoGPT includes comprehensive evaluation with **35+ benchmarks** organized into 6 categories:

### Quick Evaluation

```bash
# Run CORE benchmark evaluations during training
make ddp-train NGPUS=8 CORE_EVALS=true

# Skip evaluations for faster training
make ddp-train NGPUS=8 NO_EVALS=true
```

### Evaluation Categories

1. **📖 Reading Comprehension** (SQuAD, BoolQ, CoQA, LSAT RC, SAT English)
2. **🧠 Commonsense Reasoning** (StrategyQA, COPA, PIQA, SIQA, CommonsenseQA)
3. **🌍 World Knowledge** (MMLU, Jeopardy, TriviaQA, ARC, WikiData)
4. **🔢 Symbolic Problem Solving** (GSM8K, SVAMP, Elementary Math, Dyck Languages)
5. **📝 Language Understanding** (LAMBADA, HellaSwag, Winograd, Winogrande)
6. **🛡️ Safety** (Safety benchmarks for responsible AI)

The evaluation framework automatically rescales scores above random baseline and reports aggregate scores per category. See `resources/eval_bundle/EVAL_GAUNTLET.md` for detailed benchmark descriptions.

## 📈 Monitoring Your Training

Training metrics auto-log to **Weights & Biases**:
- 📉 Training loss (watch it go down!)
- 📊 Evaluation scores across all benchmark categories
- 📐 Learning rate schedule (that beautiful cosine decay)
- ⚡ Tokens per second (throughput metrics)
- 🎯 Gradient norms (stability indicators)

👉 View your runs at: https://wandb.ai/

## 🤖 Chat with Your Model

After training, interact with your model in a chat interface:

```bash
make chat CHECKPOINT=/path/to/checkpoint.pt

# Or directly:
uv run python scripts/chat.py --checkpoint /path/to/checkpoint.pt
```

The chat interface supports special tokens for system/user/assistant roles when using mid-trained models.

## 🛠️ Handy Commands

```bash
# 📊 Check GPU status
make gpu-status

# 🔪 Kill all GPU processes (nuclear option)
make kill-gpu

# 🔥 Keep GPUs warm for testing (useful before starting training)
make gpu-hot GPUS=0,1,2

# Or with delay (e.g., start in 2 hours)
make gpu-hot GPUS=0,1,2 DELAY=2
```

## 📚 Dataset Details

### FineWeb-Edu (Primary - Pretraining)
- 🔗 Source: [HuggingFaceFW/fineweb-edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu)
- 📦 Size: ~10 billion high-quality educational tokens
- 🎓 Quality: Filtered for educational content from Common Crawl
- 🔤 Processing: GPT-2 BPE tokenization with special tokens
- 💾 Storage: Efficient binary format (uint16) with memory mapping

### TaskMixture (Mid-Training)
- 🔗 Sources: 
  - SmolTalk: ~460K conversational examples
  - MMLU auxiliary: ~100K reasoning/knowledge examples  
  - GSM8K: ~8K math reasoning examples
- 📦 Total: ~568K specialized examples
- 🎯 Purpose: Instruction following, reasoning, and alignment
- 🔤 Processing: Formatted with chat special tokens (`<|im_start|>`, `<|im_end|>`)
- 💾 Storage: Binary format with metadata for special token handling

### OpenWebText (Legacy)
- 🔗 Source: [Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)
- 📦 Size: ~8 million documents, ~9 billion tokens
- 🔤 Processing: GPT-2 BPE tokenization
- 💾 Storage: Efficient binary format (uint16)

## 💡 Pro Tips

1. **🎮 Memory Management**: With batch_size=64 and block_size=1024, budget ~40GB VRAM per GPU
2. **🔄 Gradient Accumulation**: Auto-calculated based on GPU count and target batch size (we do the math for you!)
3. **💾 Checkpointing**: Models saved periodically during training (no progress lost!)
4. **⚡ Mixed Precision**: Uses bfloat16 for 2x speedup and 50% memory savings
5. **📊 Smart Evaluation**: Use `CORE_EVALS=true` for comprehensive benchmarking, or `NO_EVALS=true` to skip evals and train faster
6. **🎯 Two-Stage Training**: Start with FineWeb-Edu pretraining, then mid-train on TaskMixture for instruction-following
7. **🔥 GPU Warmup**: Use `make gpu-hot` with `DELAY` to reserve GPUs hours before your training run
8. **💬 Chat Ready**: Mid-trained models work great with the chat interface using special tokens

## 🔧 Troubleshooting

**😱 Out of Memory Error?**
- Turn down `batch_size` in `gpt2_model.py`
- The system auto-adjusts gradient accumulation steps (smart!)
- Try reducing evaluation batch size if OOM happens during evals

**🐌 Data Loading Slow?**
- Network filesystems don't support mmap (it's okay, we have a fallback)
- Pro tip: Copy data to local SSD for maximum zoom
- Ensure you have enough disk space (~30-50GB for FineWeb-Edu)

**🤔 Distributed Training Not Working?**
- Check NCCL installation: `python -c "import torch; print(torch.cuda.nccl.version())"`
- Verify GPUs visible: `nvidia-smi`
- Make sure all GPUs are the same model (mixed GPU types = sadness)

**📊 Evaluations Taking Too Long?**
- Use `NO_EVALS=true` to skip evaluations during training
- Or use `CORE_EVALS=true` for a focused subset of benchmarks
- Evaluations only run on rank 0 to save compute

**🎯 Mid-Training Checkpoint Not Found?**
- Ensure you've completed pretraining first or provide a valid checkpoint path
- Checkpoints are saved in the logs directory with timestamps

## 📊 About the Eval Gauntlet

The **Mosaic Eval Gauntlet v0.3.0** is a comprehensive evaluation suite with 35+ benchmarks across 6 core competencies:

### Why Multiple Benchmarks?

1. **Generalist Models Need Broad Evaluation**: LLMs can perform thousands of tasks - a handful of benchmarks can't capture their full capabilities
2. **Reduced Variance**: Aggregating across many benchmarks gives more robust performance estimates
3. **Category-Specific Insights**: Decomposed scores help understand model strengths/weaknesses for specific use cases

### Scoring Methodology

- Scores are **rescaled above random baseline** to ensure fairness
- For example: 30% accuracy on a 25% baseline = (0.30 - 0.25)/(1 - 0.25) = **6.67% above chance**
- This ensures all composite scores are normalized between 0 and 1
- Category scores are averaged, then aggregated for an overall score

See `resources/eval_bundle/EVAL_GAUNTLET.md` for complete benchmark descriptions.

## 🎓 Learning Resources

Want to understand what's happening under the hood?

- 📺 [Andrej Karpathy's GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- 📄 [Attention is All You Need](https://arxiv.org/abs/1706.03762) (the paper that started it all)
- 📚 [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)
- 🔍 [FineWeb-Edu Dataset](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) (high-quality educational data)

## 🙏 Acknowledgements

Standing on the shoulders of giants:

- Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - the OG educational GPT
- Based on OpenAI's GPT-2 architecture - thank you for open-sourcing!
- Datasets:
  - [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) by HuggingFace - high-quality educational content
  - [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) - classic web scrape dataset
  - [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) - conversational data
  - [MMLU](https://huggingface.co/datasets/cais/mmlu) - reasoning benchmarks
  - [GSM8K](https://huggingface.co/datasets/openai/gsm8k) - math reasoning
- Evaluation: [Mosaic Eval Gauntlet](https://www.mosaicml.com) by MosaicML - comprehensive benchmark suite

## 📜 License

MIT License - Go build something cool!

---

<div align="center">

**Built with ❤️ for learning and experimentation**

If this helped you understand transformers better, ⭐ star the repo!

</div>
