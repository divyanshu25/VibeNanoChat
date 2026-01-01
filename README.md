# 🚀 NanoGPT

> A clean, educational implementation of GPT-2 (124M parameters) trained on the OpenWebText dataset. Learn how transformers work by building one yourself!

## ✨ What's Inside

- **🧠 GPT-2 Architecture** (124M parameters): 12 layers, 12 attention heads, 768 dimensions of pure transformer magic
- **⚡ Distributed Training**: Scale across multiple GPUs with PyTorch DDP
- **📚 OpenWebText Dataset**: Train on ~9 billion tokens scraped from the web
- **🎯 Modern Training Stack**: Mixed precision (bfloat16), gradient clipping, cosine LR scheduling
- **📊 Experiment Tracking**: Built-in Weights & Biases integration
- **💾 Efficient Data Loading**: Memory-mapped binary files for lightning-fast I/O

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

### 3. 📊 Prepare the OpenWebText Dataset

Time to download and tokenize ~9 billion tokens of internet wisdom:

```bash
cd src/data/openwebtext
uv run python prepare.py
```

**What happens next:**
- 📥 Downloads OpenWebText dataset from HuggingFace (~54GB raw)
- 🔤 Tokenizes everything with GPT-2 BPE encoding
- 💾 Saves to `/sensei-fs/users/divgoyal/openwebtext/` (update path as needed)
- ✅ Creates: `train.bin` (~17GB, 9B tokens) and `val.bin` (~8.5MB, 4M tokens)

☕ Grab some coffee - this takes ~15-30 minutes depending on your connection!

### 4. 🔥 Train the Model

#### Single GPU Training

```bash
python src/gpt_2/ddp.py
```

#### 🚄 Multi-GPU Training (Go Fast!)

```bash
# Train with 8 GPUs (recommended for speed)
make ddp-train NGPUS=8

# Got 4 GPUs? No problem!
make ddp-train NGPUS=4

# Or go manual with torchrun:
torchrun --standalone --nproc_per_node=8 src/gpt_2/ddp.py
```

**⚙️ Training Configuration (The Sweet Spot):**
- 📦 Batch size per GPU: 64
- 📏 Sequence length: 1024 tokens
- 🎯 Total batch size: 524,288 tokens/step (2^19, perfectly balanced as all things should be)
- 🎓 Max learning rate: 6e-4 (with 715 warmup steps)
- 🏃 Total steps: 17,234 (one full epoch over 9B tokens)
- 💪 Optimizer: AdamW with weight decay 0.1
- ✂️ Gradient clipping: 1.0 (keeps those gradients in check)

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

## 📈 Monitoring Your Training

Training metrics auto-log to **Weights & Biases**:
- 📉 Training loss (watch it go down!)
- 📊 Learning rate schedule (that beautiful cosine decay)
- ⚡ Tokens per second (throughput metrics)
- 📐 Gradient norms (stability indicators)

👉 View your runs at: https://wandb.ai/

## 🛠️ Handy Commands

```bash
# 📊 Check GPU status
make gpu-status

# 🔪 Kill all GPU processes (nuclear option)
make kill-gpu

# 🔥 Keep GPUs warm for testing
make gpu-hot GPUS=0,1,2
```

## 📚 Dataset Details

**OpenWebText: The Internet in a Box**
- 🔗 Source: [Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)
- 📦 Size: ~8 million documents, ~9 billion tokens
- 🔤 Processing: GPT-2 BPE tokenization with end-of-text markers
- 💾 Storage: Efficient binary format (uint16) for blazing-fast loading

## ⚡ Performance Benchmarks

**Expected Throughput** (your mileage may vary):

| Hardware | Tokens/Second | Time per Epoch |
|----------|---------------|----------------|
| 8x A100 80GB | ~350K | ~7 hours ⏰ |
| 8x H100 80GB | ~600K | ~4 hours 🚀 |

*Training 9 billion tokens has never been this fast!*

## 💡 Pro Tips

1. **🎮 Memory Management**: With batch_size=64 and block_size=1024, budget ~40GB VRAM per GPU
2. **🔄 Gradient Accumulation**: Auto-calculated based on GPU count and target batch size (we do the math for you!)
3. **💾 Checkpointing**: Models saved periodically during training (no progress lost!)
4. **⚡ Mixed Precision**: Uses bfloat16 for 2x speedup and 50% memory savings

## 🔧 Troubleshooting

**😱 Out of Memory Error?**
- Turn down `batch_size` in `gpt2_model.py`
- The system auto-adjusts gradient accumulation steps (smart!)

**🐌 Data Loading Slow?**
- Network filesystems don't support mmap (it's okay, we have a fallback)
- Pro tip: Copy data to local SSD for maximum zoom

**🤔 Distributed Training Not Working?**
- Check NCCL installation: `python -c "import torch; print(torch.cuda.nccl.version())"`
- Verify GPUs visible: `nvidia-smi`
- Make sure all GPUs are the same model (mixed GPU types = sadness)

## 🎓 Learning Resources

Want to understand what's happening under the hood?

- 📺 [Andrej Karpathy's GPT video](https://www.youtube.com/watch?v=kCc8FmEb1nY)
- 📄 [Attention is All You Need](https://arxiv.org/abs/1706.03762) (the paper that started it all)
- 📚 [The Illustrated Transformer](http://jalammar.github.io/illustrated-transformer/)

## 🙏 Acknowledgements

Standing on the shoulders of giants:

- Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT) - the OG educational GPT
- Based on OpenAI's GPT-2 architecture - thank you for open-sourcing!
- Dataset: [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext) - internet gold

## 📜 License

MIT License - Go build something cool!

---

<div align="center">

**Built with ❤️ for learning and experimentation**

If this helped you understand transformers better, ⭐ star the repo!

</div>
