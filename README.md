# NanoGPT

A clean, educational implementation of GPT-2 (124M parameters) trained on the OpenWebText dataset. This project demonstrates transformer-based language modeling with distributed training support.

## Features

- **GPT-2 Architecture** (124M parameters): 12 layers, 12 heads, 768 embedding dimension
- **Distributed Training**: Multi-GPU support with PyTorch DDP
- **OpenWebText Dataset**: ~9B training tokens from web content
- **Modern Training**: Mixed precision (bfloat16), gradient clipping, cosine learning rate schedule
- **Experiment Tracking**: Weights & Biases integration
- **Efficient Data Loading**: Memory-mapped binary files for fast I/O

## Project Structure

```
NanoGPT/
├── src/
│   ├── gpt_2/
│   │   ├── gpt2_model.py              # GPT-2 model implementation
│   │   ├── trainer.py                 # Training loop and optimization
│   │   ├── ddp.py                     # Distributed training launcher
│   │   ├── open_webtext_dataloader.py # OpenWebText data loader
│   │   ├── attention.py               # Multi-head self-attention
│   │   ├── block.py                   # Transformer block
│   │   ├── mlp.py                     # Feedforward network
│   │   └── evaluator.py               # Model evaluation
│   └── data/
│       └── openwebtext/
│           └── prepare.py             # Dataset preprocessing script
├── Makefile                           # Convenient training commands
├── pyproject.toml                     # Dependencies
└── README.md
```

## Getting Started

### 1. Clone and Setup Environment

```bash
git clone https://github.com/yourusername/NanoGPT.git
cd NanoGPT
```

### 2. Install Dependencies

This project uses [UV](https://github.com/astral-sh/uv) for dependency management:

```bash
# Full environment setup
make environment

# Or step by step:
make uv        # Install UV
make uvlock    # Lock dependencies
make venv      # Create virtual environment
```

### 3. Prepare OpenWebText Dataset

Download and tokenize the OpenWebText dataset (~9B tokens):

```bash
cd src/data/openwebtext
uv run python prepare.py
```

This will:
- Download the OpenWebText dataset from HuggingFace
- Tokenize with GPT-2 BPE encoding
- Save to `/sensei-fs/users/divgoyal/openwebtext/` (or modify the path in `prepare.py`)
- Output: `train.bin` (~17GB, 9B tokens) and `val.bin` (~8.5MB, 4M tokens)

### 4. Train the Model

#### Single GPU Training

```bash
python src/gpt_2/ddp.py
```

#### Multi-GPU Training (Distributed Data Parallel)

```bash
# Train with 8 GPUs
make ddp-train NGPUS=8

# Or with 4 GPUs
make ddp-train NGPUS=4

# Or directly with torchrun
torchrun --standalone --nproc_per_node=8 src/gpt_2/ddp.py
```

**Training Configuration:**
- Batch size per GPU: 64
- Sequence length: 1024 tokens
- Total batch size: 524,288 tokens/step (2^19)
- Max learning rate: 6e-4
- Warmup steps: 715
- Total steps: 17,234 (1 epoch over 9B tokens)
- Optimizer: AdamW with weight decay 0.1
- Gradient clipping: 1.0

## Training Hyperparameters

### Model Config (GPT-2 124M)

```python
block_size: 1024      # Context window
vocab_size: 50257     # GPT-2 vocabulary
n_layer: 12           # Transformer blocks
n_head: 12            # Attention heads
n_embed: 768          # Embedding dimension
```

### Training Config

```python
max_learning_rate: 6e-4
min_learning_rate: 6e-5  # 10% of max
warmup_steps: 715
total_batch_size: 524288  # tokens per step
weight_decay: 0.10
gradient_clip_norm: 1.0
```

## Monitoring Training

Training metrics are logged to Weights & Biases:
- Training loss
- Learning rate schedule
- Tokens per second (throughput)
- Gradient norms

View your runs at: https://wandb.ai/

## Utilities

```bash
# Check GPU status
make gpu-status

# Kill all GPU processes
make kill-gpu

# Keep GPUs warm (for testing)
make gpu-hot GPUS=0,1,2
```

## Dataset Details

**OpenWebText**
- Source: [Skylion007/openwebtext](https://huggingface.co/datasets/Skylion007/openwebtext)
- Size: ~8M documents, ~9B tokens
- Processing: GPT-2 BPE tokenization with EOT tokens
- Storage: Binary format (uint16) for efficient loading

## Performance

Expected throughput on modern GPUs:
- A100 80GB (8x): ~350K tokens/sec
- H100 80GB (8x): ~600K tokens/sec

Total training time (1 epoch):
- 8x A100: ~7 hours
- 8x H100: ~4 hours

## Model Architecture

```
GPT-2 (124M parameters)
├── Token Embedding (50257 × 768)
├── Position Embedding (1024 × 768)
├── 12 × Transformer Block
│   ├── Layer Norm
│   ├── Multi-Head Attention (12 heads)
│   ├── Layer Norm
│   └── MLP (768 → 3072 → 768, GELU)
├── Final Layer Norm
└── Language Model Head (768 → 50257)
```

## Tips

1. **Memory Management**: With batch_size=64 and block_size=1024, each GPU needs ~40GB VRAM
2. **Gradient Accumulation**: Automatically calculated based on GPU count and target batch size
3. **Checkpointing**: Models are saved periodically during training
4. **Mixed Precision**: Uses bfloat16 for faster training and reduced memory

## Troubleshooting

**Out of Memory Error:**
- Reduce `batch_size` in `gpt2_model.py`
- The system will automatically adjust gradient accumulation steps

**Slow Data Loading:**
- The dataloader uses fallback loading on network filesystems
- For best performance, copy data to local SSD

**Distributed Training Issues:**
- Ensure NCCL is properly installed
- Check that all GPUs are visible: `nvidia-smi`

## Acknowledgements

- Inspired by [Andrej Karpathy's nanoGPT](https://github.com/karpathy/nanoGPT)
- Based on OpenAI's GPT-2 architecture
- Dataset: [OpenWebText](https://huggingface.co/datasets/Skylion007/openwebtext)

## License

MIT License
