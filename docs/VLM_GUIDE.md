# Adding Vision to NanoGPT: A Practical Guide

## The Big Question

**"I have a trained NanoGPT model. Can I add images to it without retraining everything from scratch?"**

**YES.** And it's actually pretty simple. This is how modern vision-language models like LLaVA work.

---

## The Core Idea (ELI5)

Your NanoGPT already understands language. You're going to:

1. **Plug in a "vision translator"** (CLIP) that converts images into vectors
2. **Add a tiny bridge** (~2M parameters) that translates these vectors into your model's "language"
3. **Fine-tune everything together** on image-text pairs

That's it. No retraining from scratch. No billions of image-text pairs. Just standard supervised fine-tuning (SFT) with images.

**Time needed:** 10-20 hours on 4-8 GPUs (not 500+ hours of pretraining)

**Data needed:** 50K-150K instruction examples with images (not billions of pairs)

---

## How Modern VLMs Actually Work

Let's look at LLaVA, which achieves excellent results with this simple approach:

```
┌─────────────────────────────────────────────────────────────┐
│                    LLaVA Architecture                        │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Image                               Text                   │
│   │                                   │                      │
│   ▼                                   │                      │
│ ┌──────────┐                         │                      │
│ │   CLIP   │ (frozen, pretrained)    │                      │
│ │  ViT-L   │                          │                      │
│ └──────────┘                         │                      │
│   │                                   │                      │
│   │ [257 x 1024 features]            │                      │
│   ▼                                   │                      │
│ ┌──────────┐                         │                      │
│ │ Project  │ (NEW, ~2M params)       │                      │
│ └──────────┘                         │                      │
│   │                                   │                      │
│   │ [257 x n_embed]                  │                      │
│   ▼                                   ▼                      │
│   └──────────── Concat ──────────► [combined tokens]       │
│                                       │                      │
│                                       ▼                      │
│                                  ┌─────────┐                │
│                                  │   GPT   │ (fine-tune)    │
│                                  │  Model  │                │
│                                  └─────────┘                │
│                                       │                      │
│                                       ▼                      │
│                                  [output text]              │
└─────────────────────────────────────────────────────────────┘
```

**Key insight:** The image becomes just another sequence of tokens. Your GPT model doesn't care if a token came from text or an image - it's all just vectors.

---

## What You Actually Need to Do

### Step 1: Use Your Existing Checkpoint

```python
# Load any of your existing checkpoints:
checkpoint_path = "checkpoints/pretrain/model_final.pt"      # Works!
# or:
checkpoint_path = "checkpoints/midtrain/model_50000.pt"      # Works!
# or:
checkpoint_path = "checkpoints/sft/model_final.pt"           # Works!
```

Your model already knows language. We're just teaching it to "see."

### Step 2: Add Vision Components

```python
# config.py - add these settings
config.use_vision = True
config.vision_encoder_name = "openai/clip-vit-large-patch14"
config.vision_encoder_hidden_size = 1024  # ViT-L dimension
config.freeze_vision_encoder = True       # Keep CLIP frozen
```

This adds:
- **CLIP encoder** (frozen, ~300M params) - already trained, we're not changing it
- **Projection layer** (trainable, ~2M params) - maps CLIP → GPT space

### Step 3: Train on Image-Text Pairs

```bash
# Prepare dataset (LLaVA-Instruct-150K is a good starting point)
python data/multimodal/prepare_dataset.py

# Train (10-20 hours on 4-8 GPUs)
python scripts/train_vlm_sft.py
```

That's it. You now have a vision-language model.

---

## Training Strategies: Pick Your Path

### Option 1: Direct SFT (RECOMMENDED)

**What it is:** Load your checkpoint, add vision, train on instruction data with images.

```
Your NanoGPT → + CLIP (frozen) + Projection (new) → Train 5-10K steps → VLM ✓
```

**When to use:** You want results fast and don't need absolute maximum quality.

**Time:** 10-20 hours (4-8 GPUs)  
**Data:** 50K-150K instruction pairs (e.g., LLaVA-Instruct)  
**Quality:** Excellent (this is what most VLMs do)

**Training config:**
```python
config.batch_size = 8              # Images use more memory
config.max_learning_rate = 2e-5    # Lower LR for fine-tuning
config.num_iterations = 5000       # Typical for SFT
```

### Option 2: Two-Stage (LLaVA-Style)

**What it is:** First align vision↔language with captions, then do instruction tuning.

```
Stage 1: Train projection only (freeze everything else) → 5-8 hours
         Data: 500K-5M captions (COCO, CC3M)
         
Stage 2: Train projection + LLM (freeze CLIP) → 10-20 hours
         Data: 50K-150K instructions
```

**When to use:** You have caption data and want that extra 2-5% quality boost.

**Time:** 15-30 hours total  
**Quality:** Slightly better than Option 1

**Stage 1 config:**
```python
# Freeze LLM, train only projection
for name, param in model.named_parameters():
    if 'vision_projector' not in name:
        param.requires_grad = False
```

### Option 3: Full Multimodal Pretraining (NOT RECOMMENDED)

**What it is:** Pretrain from scratch with billions of image-text pairs.

**When to use:** You're Google/Meta and have unlimited compute.

**Time:** 500+ hours  
**Cost:** $$$$$  
**Quality gain over Option 2:** Marginal (~1-3%)

**Our take:** Don't do this unless you have a very specific reason. Modern VLMs (LLaVA, InstructBLIP, LLaVA-1.5) all use Option 1 or 2.

---

## Quick Comparison

| Approach | Time | Data Needed | Quality | When to Use |
|----------|------|-------------|---------|-------------|
| **Direct SFT** | 10-20 hrs | 50K-150K instructions | ⭐⭐⭐⭐ | Default choice |
| **Two-Stage** | 15-30 hrs | 500K captions + 50K instructions | ⭐⭐⭐⭐⭐ | Want max quality |
| **Full Pretrain** | 500+ hrs | Billions of pairs | ⭐⭐⭐⭐⭐ | Research only |

**Bottom line:** Start with Direct SFT. It's 95% as good as Two-Stage in 60% of the time.

---

## Implementation Guide

### 1. Install Dependencies

Add to `pyproject.toml`:

```toml
dependencies = [
    "transformers>=4.36.0",
    "pillow>=10.0.0",
    "open-clip-torch>=2.24.0",
    "torchvision>=0.16.0",
]
```

```bash
uv sync
```

### 2. Update Config

```python
# src/gpt_2/config.py
# Add these fields to GPTConfig class:

# Vision settings
use_vision: bool = False
vision_encoder_name: str = "openai/clip-vit-large-patch14"
vision_encoder_hidden_size: int = 1024      # 1024 for ViT-L, 768 for ViT-B
vision_num_tokens: int = 257                # Number of image patches
vision_projection_type: str = "linear"      # "linear" or "mlp"
freeze_vision_encoder: bool = True
image_size: int = 224

# Data
image_text_data_dir: str = "/path/to/dataset"
max_images_per_sequence: int = 1
```

### 3. Create Vision Encoder

Create `src/gpt_2/vision_encoder.py`:

```python
import torch
import torch.nn as nn
from transformers import CLIPVisionModel, CLIPImageProcessor


class VisionEncoder(nn.Module):
    """Frozen CLIP vision encoder."""
    
    def __init__(self, config):
        super().__init__()
        self.config = config
        
        # Load pretrained CLIP
        self.vision_model = CLIPVisionModel.from_pretrained(
            config.vision_encoder_name
        )
        
        # Freeze if specified
        if config.freeze_vision_encoder:
            for param in self.vision_model.parameters():
                param.requires_grad = False
            self.vision_model.eval()
        
        # Image preprocessor
        self.image_processor = CLIPImageProcessor.from_pretrained(
            config.vision_encoder_name
        )
        
    def forward(self, pixel_values):
        """
        Args:
            pixel_values: (B, C, H, W)
        Returns:
            vision_features: (B, num_tokens, hidden_size)
        """
        if self.config.freeze_vision_encoder:
            with torch.no_grad():
                outputs = self.vision_model(pixel_values)
        else:
            outputs = self.vision_model(pixel_values)
        
        # Get all patch embeddings + CLS token
        return outputs.last_hidden_state  # (B, 257, 1024) for ViT-L


class VisionProjector(nn.Module):
    """Maps CLIP features to GPT embedding space."""
    
    def __init__(self, config):
        super().__init__()
        
        if config.vision_projection_type == "linear":
            self.projection = nn.Linear(
                config.vision_encoder_hidden_size,
                config.n_embed,
                bias=True
            )
        elif config.vision_projection_type == "mlp":
            # 2-layer MLP for more capacity
            self.projection = nn.Sequential(
                nn.Linear(config.vision_encoder_hidden_size, config.n_embed * 2),
                nn.GELU(),
                nn.Linear(config.n_embed * 2, config.n_embed),
            )
        else:
            raise ValueError(f"Unknown projection: {config.vision_projection_type}")
    
    def forward(self, vision_features):
        """
        Args:
            vision_features: (B, num_tokens, vision_hidden_size)
        Returns:
            projected: (B, num_tokens, n_embed)
        """
        return self.projection(vision_features)


class VisionLanguageEncoder(nn.Module):
    """Complete vision-language encoder."""
    
    def __init__(self, config):
        super().__init__()
        self.vision_encoder = VisionEncoder(config)
        self.vision_projector = VisionProjector(config)
        
    def forward(self, pixel_values):
        vision_features = self.vision_encoder(pixel_values)
        image_embeddings = self.vision_projector(vision_features)
        return image_embeddings
    
    def get_image_processor(self):
        return self.vision_encoder.image_processor
```

### 4. Modify GPT Model

Update `src/gpt_2/gpt2_model.py`:

**A. Add import:**

```python
from gpt_2.vision_encoder import VisionLanguageEncoder
```

**B. Add to `__init__`:**

```python
def __init__(self, config):
    super().__init__()
    self.config = config
    
    # ... existing transformer setup ...
    
    # Add vision encoder if enabled
    if config.use_vision:
        self.vision_encoder = VisionLanguageEncoder(config)
    else:
        self.vision_encoder = None
    
    # ... rest of init ...
```

**C. Add multimodal forward method:**

```python
def forward_multimodal(self, idx, pixel_values=None, targets=None, loss_reduction="mean"):
    """
    Forward pass with optional images.
    
    Args:
        idx: Text tokens (B, T)
        pixel_values: Images (B, num_images, C, H, W)
        targets: Target tokens for loss
        
    Returns:
        logits, loss
    """
    B, T = idx.size()
    
    # Text embeddings
    tok_emb = self.transformer.wte(idx)  # (B, T, n_embed)
    
    # Add image embeddings if provided
    if pixel_values is not None and self.vision_encoder is not None:
        # Process images
        B_img, num_images, C, H, W = pixel_values.shape
        pixel_values_flat = pixel_values.view(B_img * num_images, C, H, W)
        
        # Get image embeddings
        image_embeddings = self.vision_encoder(pixel_values_flat)
        num_img_tokens = image_embeddings.shape[1]
        
        # Reshape: (B, num_images, num_tokens, n_embed)
        image_embeddings = image_embeddings.view(B_img, num_images, num_img_tokens, self.config.n_embed)
        
        # Concatenate: [image tokens, text tokens] for each sample
        embeddings_list = []
        for b in range(B):
            img_embs = image_embeddings[b].reshape(-1, self.config.n_embed)  # Flatten images
            txt_emb = tok_emb[b]
            combined = torch.cat([img_embs, txt_emb], dim=0)
            embeddings_list.append(combined)
        
        x = torch.stack(embeddings_list, dim=0)
        T_total = x.shape[1]
    else:
        x = tok_emb
        T_total = T
    
    # Position embeddings
    pos = torch.arange(0, T_total, dtype=torch.long, device=idx.device)
    pos_emb = self.transformer.wpe(pos)
    x = x + pos_emb
    
    # Transformer blocks
    for block in self.transformer.h:
        x = block(x)
    
    x = self.transformer.ln_f(x)
    logits = self.lm_head(x)
    
    # Loss computation
    loss = None
    if targets is not None:
        if pixel_values is not None:
            # Only compute loss on text tokens (skip image tokens)
            num_img_tokens_total = num_img_tokens * num_images
            logits_for_loss = logits[:, num_img_tokens_total:, :]
            targets_for_loss = targets
        else:
            logits_for_loss = logits
            targets_for_loss = targets
        
        loss = F.cross_entropy(
            logits_for_loss.reshape(-1, logits_for_loss.size(-1)),
            targets_for_loss.reshape(-1),
            ignore_index=-1,
            reduction=loss_reduction,
        )
    
    return logits, loss
```

### 5. Create Multimodal Dataloader

Create `src/dataloaders/multimodal_dataloader.py`:

```python
import json
from pathlib import Path
import torch
from PIL import Image
from torch.utils.data import Dataset, DataLoader


class MultimodalDataset(Dataset):
    """
    Dataset for image-text pairs.
    
    Expected format (JSONL):
    {
        "image": "path/to/image.jpg",
        "text": "Caption or instruction",
        "conversation": [  # Optional for instruction tuning
            {"role": "user", "content": "What's in this image?"},
            {"role": "assistant", "content": "The image shows..."}
        ]
    }
    """
    
    def __init__(self, data_dir, split="train", max_length=2048, 
                 image_processor=None, tokenizer=None):
        self.data_dir = Path(data_dir)
        self.split = split
        self.max_length = max_length
        self.image_processor = image_processor
        self.tokenizer = tokenizer
        
        # Load metadata
        metadata_file = self.data_dir / f"{split}.jsonl"
        self.samples = []
        
        with open(metadata_file, 'r') as f:
            for line in f:
                self.samples.append(json.loads(line))
        
        print(f"Loaded {len(self.samples)} {split} samples")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        
        # Load image
        image_path = self.data_dir / sample['image']
        image = Image.open(image_path).convert('RGB')
        
        if self.image_processor is not None:
            pixel_values = self.image_processor(
                images=image, 
                return_tensors="pt"
            )['pixel_values'][0]
        else:
            pixel_values = None
        
        # Format text
        if 'conversation' in sample:
            text = self._format_conversation(sample['conversation'])
        else:
            text = sample['text']
        
        # Tokenize
        if self.tokenizer is not None:
            tokens = self.tokenizer.encode(text, allowed_special="all")
            if len(tokens) > self.max_length:
                tokens = tokens[:self.max_length]
            tokens = torch.tensor(tokens, dtype=torch.long)
        else:
            tokens = None
        
        return {
            'pixel_values': pixel_values,
            'input_ids': tokens,
            'text': text,
        }
    
    def _format_conversation(self, conversation):
        """Format conversation into text."""
        formatted = ""
        for turn in conversation:
            role = turn['role']
            content = turn['content']
            if role == 'user':
                formatted += f"User: {content}\n"
            elif role == 'assistant':
                formatted += f"Assistant: {content}\n"
        return formatted


def collate_multimodal(batch):
    """Collate function with padding."""
    pixel_values = torch.stack([item['pixel_values'] for item in batch])
    
    # Pad text sequences
    input_ids = [item['input_ids'] for item in batch]
    max_len = max(len(ids) for ids in input_ids)
    
    padded_input_ids = []
    padded_targets = []
    
    for ids in input_ids:
        padding_length = max_len - len(ids)
        # Pad with -1 (ignored in loss)
        padded_ids = torch.cat([
            ids,
            torch.full((padding_length,), -1, dtype=torch.long)
        ])
        padded_input_ids.append(padded_ids)
        
        # Targets are shifted by 1
        targets = torch.cat([
            ids[1:],
            torch.tensor([-1], dtype=torch.long),
            torch.full((padding_length,), -1, dtype=torch.long)
        ])
        padded_targets.append(targets)
    
    input_ids = torch.stack(padded_input_ids)
    targets = torch.stack(padded_targets)
    
    return {
        'pixel_values': pixel_values.unsqueeze(1),  # Add num_images dim
        'input_ids': input_ids,
        'targets': targets,
    }


class MultimodalDataloader:
    """Wrapper for trainer compatibility."""
    
    def __init__(self, config, split='train', tokenizer=None, image_processor=None):
        self.config = config
        self.split = split
        
        self.dataset = MultimodalDataset(
            data_dir=config.image_text_data_dir,
            split=split,
            max_length=config.block_size,
            image_processor=image_processor,
            tokenizer=tokenizer,
        )
        
        self.dataloader = DataLoader(
            self.dataset,
            batch_size=config.batch_size,
            shuffle=(split == 'train'),
            collate_fn=collate_multimodal,
            num_workers=4,
            pin_memory=True,
        )
        
        self.iterator = iter(self.dataloader)
    
    def get_batch(self, split=None):
        """Get next batch (trainer-compatible interface)."""
        try:
            batch = next(self.iterator)
        except StopIteration:
            self.iterator = iter(self.dataloader)
            batch = next(self.iterator)
        
        return batch['input_ids'], batch['targets'], batch['pixel_values']
```

### 6. Update Trainer

Modify `src/gpt_2/trainer.py`:

**A. Add import:**

```python
from dataloaders.multimodal_dataloader import MultimodalDataloader
```

**B. Initialize multimodal dataloader in `__init__`:**

```python
# Add in __init__ after regular dataloader setup:
if self.config.use_vision:
    print("Initializing multimodal dataloader...")
    image_processor = self.model.vision_encoder.get_image_processor()
    
    self.train_loader = MultimodalDataloader(
        config=self.config,
        split='train',
        tokenizer=self.enc,
        image_processor=image_processor,
    )
    
    self.val_loader = MultimodalDataloader(
        config=self.config,
        split='val',
        tokenizer=self.enc,
        image_processor=image_processor,
    )
```

**C. Update training loop:**

```python
# In train() method, replace:
# logits, loss = self.model(x, y)

# With:
if self.config.use_vision:
    x, y, pixel_values = batch  # Unpack multimodal batch
    x, y = x.to(self.device), y.to(self.device)
    pixel_values = pixel_values.to(self.device)
    
    logits, loss = self.model.forward_multimodal(
        x, 
        pixel_values=pixel_values,
        targets=y,
        loss_reduction="mean"
    )
else:
    logits, loss = self.model(x, y)
```

### 7. Prepare Dataset

Create `data/multimodal/prepare_dataset.py`:

```python
"""Prepare multimodal dataset from HuggingFace."""

import json
from pathlib import Path
from datasets import load_dataset


def prepare_llava_instruct(output_dir, max_samples=None):
    """
    Prepare LLaVA-Instruct-150K dataset.
    
    This dataset contains instruction-following examples with images.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("Loading LLaVA-Instruct dataset...")
    dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K")
    
    train_split = dataset['train']
    if max_samples:
        train_split = train_split.select(range(min(max_samples, len(train_split))))
    
    train_jsonl = output_dir / "train.jsonl"
    images_dir = output_dir / "images"
    images_dir.mkdir(exist_ok=True)
    
    with open(train_jsonl, 'w') as f:
        for idx, sample in enumerate(train_split):
            # Save image
            image = sample['image']
            image_filename = f"train_{idx:06d}.jpg"
            image_path = images_dir / image_filename
            image.save(image_path)
            
            # Format conversation
            conversations = sample['conversations']
            formatted_conv = []
            for turn in conversations:
                formatted_conv.append({
                    'role': 'user' if turn['from'] == 'human' else 'assistant',
                    'content': turn['value']
                })
            
            # Write metadata
            metadata = {
                'image': str(Path('images') / image_filename),
                'conversation': formatted_conv,
            }
            f.write(json.dumps(metadata) + '\n')
            
            if (idx + 1) % 1000 == 0:
                print(f"Processed {idx + 1} samples...")
    
    print(f"\nDataset prepared: {train_jsonl}")
    print(f"Total samples: {len(train_split)}")


if __name__ == "__main__":
    prepare_llava_instruct(
        output_dir="/path/to/output/llava_dataset",
        max_samples=50000  # Start with 50K for faster iteration
    )
```

### 8. Training Script

Create `scripts/train_vlm_sft.py`:

```python
"""
Vision-Language Model Training Script

This shows how to:
1. Load your existing pretrained checkpoint
2. Add vision components
3. Train on multimodal instruction data
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT
from gpt_2.trainer import Trainer


def main():
    # Configure for VLM
    config = GPTConfig()
    
    # Enable vision
    config.use_vision = True
    config.vision_encoder_name = "openai/clip-vit-large-patch14"
    config.vision_encoder_hidden_size = 1024
    config.vision_num_tokens = 257
    config.vision_projection_type = "linear"
    config.freeze_vision_encoder = True
    
    # Data path
    config.image_text_data_dir = "/path/to/llava_dataset"
    
    # Training params (adjusted for multimodal)
    config.batch_size = 8                  # Reduced for memory
    config.block_size = 1024               # Shorter context
    config.max_learning_rate = 2e-5        # Lower LR for fine-tuning
    config.num_iterations = 5000
    config.weight_decay = 0.01
    
    # SFT settings
    config.sft_training = True
    config.lr_warmup_ratio_sft = 0.1
    config.min_lr_ratio = 0.1
    
    # Load existing checkpoint
    pretrained_checkpoint_path = "checkpoints/pretrain/model_final.pt"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    print("=" * 70)
    print("Initializing VLM model...")
    print("=" * 70)
    
    model = GPT(config)
    
    # Load pretrained weights (text model only)
    if os.path.exists(pretrained_checkpoint_path):
        print(f"\n✓ Loading checkpoint: {pretrained_checkpoint_path}")
        checkpoint = torch.load(pretrained_checkpoint_path, map_location='cpu')
        
        # Load with strict=False (vision components are new)
        missing_keys, unexpected_keys = model.load_state_dict(
            checkpoint['model'], 
            strict=False
        )
        
        print(f"  - Loaded text model weights")
        print(f"  - New vision components: {len(missing_keys)} parameters")
        print(f"  - Vision projection will be trained from scratch")
    else:
        print(f"\n⚠ Warning: Checkpoint not found at {pretrained_checkpoint_path}")
        print("  Starting from random initialization (not recommended!)")
    
    model.to(device)
    
    # Print stats
    print("\n" + "=" * 70)
    print("Model Statistics")
    print("=" * 70)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters:      {total_params:>12,}")
    print(f"Trainable parameters:  {trainable_params:>12,}")
    print(f"Trainable %:           {100 * trainable_params / total_params:>11.1f}%")
    
    # Initialize trainer
    print("\n" + "=" * 70)
    print("Starting Training")
    print("=" * 70)
    
    trainer = Trainer(
        ddp=False,
        ddp_rank=0,
        ddp_local_rank=0,
        ddp_world_size=1,
        master_process=True,
        device=device,
        run_evals=True,
        run_core_evals=False,
        run_chatcore_evals=True,
        mid_training=False,
        sft_training=True,
        checkpoint_path=None,
        checkpoint_dir="checkpoints/vlm_sft",
    )
    
    trainer.train()
    
    print("\n" + "=" * 70)
    print("✓ Training completed!")
    print("=" * 70)


if __name__ == "__main__":
    main()
```

Run training:

```bash
# Step 1: Prepare data
python data/multimodal/prepare_dataset.py

# Step 2: Train VLM
python scripts/train_vlm_sft.py
```

---

## Testing Your VLM

Create `scripts/test_vlm.py`:

```python
"""Simple VLM inference test."""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

import torch
from PIL import Image
from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT
from gpt_2.utils import get_custom_tokenizer


def generate_from_image(model, image_path, prompt, max_length=256):
    """Generate text from image + prompt."""
    device = next(model.parameters()).device
    
    # Load image
    image = Image.open(image_path).convert('RGB')
    image_processor = model.vision_encoder.get_image_processor()
    pixel_values = image_processor(images=image, return_tensors="pt")['pixel_values']
    pixel_values = pixel_values.unsqueeze(0).to(device)  # (1, 1, C, H, W)
    
    # Tokenize prompt
    enc, _ = get_custom_tokenizer()
    tokens = enc.encode(prompt, allowed_special="all")
    input_ids = torch.tensor([tokens], dtype=torch.long).to(device)
    
    # Generate (simple greedy decoding)
    model.eval()
    with torch.no_grad():
        for _ in range(max_length):
            logits, _ = model.forward_multimodal(input_ids, pixel_values=pixel_values)
            next_token = torch.argmax(logits[0, -1, :]).item()
            
            if next_token == enc.eot_token:
                break
            
            input_ids = torch.cat([
                input_ids,
                torch.tensor([[next_token]], device=device)
            ], dim=1)
        
        generated_text = enc.decode(input_ids[0].tolist())
        return generated_text


def main():
    # Load config and model
    config = GPTConfig()
    config.use_vision = True
    config.vision_encoder_name = "openai/clip-vit-large-patch14"
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = GPT(config)
    
    # Load trained checkpoint
    checkpoint_path = "checkpoints/vlm_sft/model_5000.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(checkpoint['model'])
    model.to(device)
    model.eval()
    
    # Test
    image_path = "test_images/example.jpg"
    prompt = "User: Describe this image in detail.\nAssistant:"
    
    print(f"Image: {image_path}")
    print(f"Prompt: {prompt}\n")
    print("Generated:")
    
    output = generate_from_image(model, image_path, prompt)
    print(output)


if __name__ == "__main__":
    main()
```

---

## Performance Tips

### Memory Optimization

**Problem:** Out of memory errors

**Solutions:**
```python
# 1. Reduce batch size
config.batch_size = 4  # Instead of 8

# 2. Shorter sequences
config.block_size = 512  # Instead of 1024

# 3. Smaller vision encoder
config.vision_encoder_name = "openai/clip-vit-base-patch16"  # 768-dim instead of 1024

# 4. Gradient accumulation
config.gradient_accumulation_steps = 4
config.batch_size = 2  # Effective batch size = 2 * 4 = 8
```

### Training Speed

**CLIP is frozen:** Forward pass uses `torch.no_grad()`, so it's fast.

**Data loading:** Use multiple workers:
```python
DataLoader(dataset, num_workers=4, pin_memory=True)
```

**Mixed precision:** Enable bf16/fp16 if your GPU supports it:
```python
config.use_amp = True
```

### Quality Tips

1. **Start from a good text checkpoint:** The better your text model, the better your VLM
2. **Learning rate matters:** Too high (>1e-4) can destroy text knowledge, too low (<1e-6) won't learn vision
3. **Data quality > quantity:** 50K high-quality examples beats 500K noisy ones
4. **Two-stage helps:** If direct SFT isn't working, try the two-stage approach

---

## Datasets

### Recommended Starting Datasets

**For Direct SFT:**
- **LLaVA-Instruct-150K** - 150K instruction examples, ideal for getting started
- **LLaVA-1.5-mix** - 665K mixed instruction data, better quality
- **ShareGPT4V** - GPT-4V generated instructions, highest quality

**For Two-Stage (Stage 1):**
- **COCO Captions** - 500K image-caption pairs, clean and diverse
- **Conceptual Captions (CC3M)** - 3M web-scraped pairs, noisy but large
- **Combination** - COCO + subset of CC3M = ~1-2M pairs

### Loading from HuggingFace

```python
from datasets import load_dataset

# LLaVA-Instruct
dataset = load_dataset("liuhaotian/LLaVA-Instruct-150K")

# COCO
dataset = load_dataset("HuggingFaceM4/COCO")

# Conceptual Captions
dataset = load_dataset("conceptual_captions")
```

---

## Troubleshooting

### Issue: "Dimension mismatch in projection layer"

**Cause:** `vision_encoder_hidden_size` doesn't match your CLIP model.

**Fix:**
```python
# For ViT-L
config.vision_encoder_hidden_size = 1024

# For ViT-B
config.vision_encoder_hidden_size = 768
```

### Issue: "Model generates text gibberish for images"

**Possible causes:**
1. Forgot to load pretrained checkpoint → Start from trained text model
2. Learning rate too high → Try 1e-5 or 2e-5
3. Not enough training → Need at least 2-3K iterations
4. Wrong image preprocessing → Verify CLIP processor is used correctly

### Issue: "Training loss not decreasing"

**Check:**
1. Is the projection layer actually trainable? (print `param.requires_grad`)
2. Is the vision encoder frozen? (should be)
3. Are images loading correctly? (visualize a batch)
4. Is learning rate too low? (try 2e-5)

### Issue: "Slow training (< 1 it/sec)"

**Optimizations:**
1. Verify vision encoder is frozen (should use `no_grad`)
2. Use `num_workers > 0` in DataLoader
3. Enable `pin_memory=True`
4. Preprocess images offline if possible
5. Use mixed precision training

---

## FAQ

### Q: Can I use this with my custom tokenizer?

**A:** Yes! The vision encoder is independent. Just make sure your tokenizer handles any special tokens you add (e.g., `<image>`).

### Q: What if I don't have 4-8 GPUs?

**A:** You can train on 1 GPU, it'll just take longer (~40-80 hours instead of 10-20). Or use smaller models:
- Smaller NanoGPT (256M params instead of 560M)
- ViT-B instead of ViT-L
- Batch size 2-4 with gradient accumulation

### Q: Can I add vision to a model that's already been SFT'd?

**A:** Yes! Load your SFT checkpoint and continue training with images. Your instruction-following ability will transfer to the visual domain.

### Q: How much VRAM do I need?

**A:** Rough estimates:
- 1x A100 (40GB): Comfortable with batch_size=8
- 1x V100 (32GB): Works with batch_size=4-6
- 1x RTX 4090 (24GB): Works with batch_size=2-4
- Multiple smaller GPUs: Use DDP

### Q: Does this work with other architectures (not CLIP)?

**A:** Yes! You can swap CLIP for:
- **SigLIP** (better than CLIP)
- **DINOv2** (better semantic understanding)
- **EVA-CLIP** (larger, more powerful)

Just change `vision_encoder_name` and update the hidden size.

### Q: Can I train the vision encoder too?

**A:** You can, but it's not recommended. Frozen CLIP works very well and training it:
- Requires much more memory
- Needs more data (billions of pairs)
- Gives marginal improvements
- Risks degrading pretrained vision features

### Q: Should I use linear or MLP projection?

**A:** Start with `linear` (simpler, faster). If quality isn't great, try `mlp` (more capacity). The difference is usually small (~1-2%).

---

## Summary: What You Actually Need

✅ **DO THIS:**
1. Load your existing NanoGPT checkpoint
2. Add frozen CLIP + projection layer
3. Train on 50K-150K instruction pairs with images
4. Train for 5-10K iterations (~10-20 hours)

❌ **DON'T DO THIS:**
1. Retrain your text model from scratch
2. Re-run pretraining with billions of image-text pairs
3. Re-run midtraining with images
4. Train the CLIP encoder
5. Overthink it

**Time investment:** 10-30 hours  
**Data needed:** 50K-150K examples  
**Expected quality:** Excellent (LLaVA-level)

---

## References

- **LLaVA Paper:** Liu et al., "Visual Instruction Tuning", NeurIPS 2023
  - ArXiv: https://arxiv.org/abs/2304.08485
  - GitHub: https://github.com/haotian-liu/LLaVA
- **CLIP:** Radford et al., "Learning Transferable Visual Models", ICML 2021
- **LLaVA-1.5:** Improved version with better data mix
- **InstructBLIP:** Similar approach with BLIP-2 as vision encoder

---

## Next Steps

1. ✅ Prepare your dataset (start with LLaVA-Instruct-150K)
2. ✅ Run training script (start with 50K samples for quick iteration)
3. ✅ Evaluate on a few examples manually
4. ✅ Scale up if results look good
5. ✅ Share your results!

Good luck! 🚀
