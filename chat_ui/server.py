#!/usr/bin/env python3
"""
FastAPI Chat Server for NanoGPT

Serves a web interface to chat with trained checkpoints.

Usage:
    python server.py
"""

import os
import sys
from pathlib import Path
from typing import List, Optional

# Add gpt_2 to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.append(src_dir)

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT
from gpt_2.utils import get_custom_tokenizer, load_checkpoint

app = FastAPI(title="NanoGPT Chat Server")

# Global state
CHECKPOINT_DIR = "/sensei-fs/users/divgoyal/nanogpt/midtrain_checkpoints"
model = None
tokenizer = None
device = None
current_checkpoint = None
conversation_history = []


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_model(checkpoint_path: str):
    """Load a checkpoint into the global model."""
    global model, tokenizer, device, current_checkpoint

    if device is None:
        device = get_device()
        print(f"🖥️  Device: {device}")

    if tokenizer is None:
        tokenizer, _ = get_custom_tokenizer()

    print(f"🔧 Loading checkpoint: {checkpoint_path}")
    model = GPT(GPTConfig())
    model.to(device)
    load_checkpoint(checkpoint_path, model, device, optimizer=None, master_process=True)
    model.eval()
    current_checkpoint = checkpoint_path
    print(f"✅ Loaded: {os.path.basename(checkpoint_path)}")


def format_chat_prompt(messages):
    """Format conversation history into the chat prompt format."""
    prompt = "<|bos|>"
    for msg in messages:
        if msg["role"] == "user":
            prompt += f"<|user_start|>{msg['content']}<|user_end|>"
        elif msg["role"] == "assistant":
            prompt += f"<|assistant_start|>{msg['content']}<|assistant_end|>"
    prompt += "<|assistant_start|>"
    return prompt


def generate_response(
    prompt: str, max_new_tokens: int = 256, temperature: float = 0.8, top_k: int = 50
):
    """Generate a response from the model given a prompt."""
    if model is None:
        raise ValueError("No model loaded")

    model.eval()

    tokens = tokenizer.encode(prompt, allowed_special="all")
    tokens = torch.tensor(tokens, dtype=torch.long, device=device).unsqueeze(0)
    prompt_length = tokens.size(1)

    stop_tokens = [
        tokenizer.encode("<|assistant_end|>", allowed_special="all")[0],
        tokenizer.encode("<|endoftext|>", allowed_special="all")[0],
        tokenizer.encode("<|user_start|>", allowed_special="all")[0],
        tokenizer.encode("<|user_end|>", allowed_special="all")[0],
    ]

    with torch.no_grad():
        for _ in range(max_new_tokens):
            if tokens.size(1) > model.config.block_size:
                tokens = tokens[:, -model.config.block_size :]

            logits, _ = model(tokens)
            logits = logits[:, -1, :] / temperature

            # Top-k sampling
            v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
            logits[logits < v[:, [-1]]] = float("-inf")

            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            tokens = torch.cat([tokens, next_token], dim=1)

            if next_token.item() in stop_tokens:
                break

    generated_tokens = tokens[0, prompt_length:].tolist()
    raw_response = tokenizer.decode(generated_tokens)

    # Clean up special tokens
    for token_str in [
        "<|assistant_end|>",
        "<|endoftext|>",
        "<|user_start|>",
        "<|user_end|>",
        "<|assistant_start|>",
        "<|bos|>",
    ]:
        raw_response = raw_response.replace(token_str, "")

    return raw_response.strip()


# API Models
class ChatRequest(BaseModel):
    message: str
    max_tokens: Optional[int] = 256
    temperature: Optional[float] = 0.8
    top_k: Optional[int] = 50


class ChatResponse(BaseModel):
    response: str
    conversation: List[dict]


class CheckpointRequest(BaseModel):
    checkpoint: str


# API Endpoints
@app.get("/")
async def root():
    """Serve the chat UI."""
    html_path = Path(__file__).parent / "index.html"
    if html_path.exists():
        return HTMLResponse(content=html_path.read_text())
    return HTMLResponse(content="<h1>Chat UI not found. Please create index.html</h1>")


@app.get("/api/checkpoints")
async def list_checkpoints():
    """List all available checkpoints."""
    try:
        checkpoint_dir = Path(CHECKPOINT_DIR)
        if not checkpoint_dir.exists():
            return JSONResponse(
                content={
                    "checkpoints": [],
                    "error": f"Directory not found: {CHECKPOINT_DIR}",
                }
            )

        checkpoints = sorted(
            [str(f.relative_to(checkpoint_dir)) for f in checkpoint_dir.glob("**/*.pt")]
        )

        return JSONResponse(
            content={
                "checkpoints": checkpoints,
                "current": (
                    os.path.basename(current_checkpoint) if current_checkpoint else None
                ),
            }
        )
    except Exception as e:
        return JSONResponse(content={"checkpoints": [], "error": str(e)})


@app.post("/api/load_checkpoint")
async def load_checkpoint_endpoint(request: CheckpointRequest):
    """Load a specific checkpoint."""
    try:
        checkpoint_path = os.path.join(CHECKPOINT_DIR, request.checkpoint)
        if not os.path.exists(checkpoint_path):
            raise HTTPException(
                status_code=404, detail=f"Checkpoint not found: {request.checkpoint}"
            )

        load_model(checkpoint_path)

        return JSONResponse(
            content={
                "success": True,
                "checkpoint": request.checkpoint,
                "message": f"Loaded {request.checkpoint}",
            }
        )
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Send a message and get a response."""
    global conversation_history

    if model is None:
        raise HTTPException(
            status_code=400,
            detail="No checkpoint loaded. Please load a checkpoint first.",
        )

    try:
        # Add user message to history
        conversation_history.append({"role": "user", "content": request.message})

        # Generate prompt from conversation history
        prompt = format_chat_prompt(conversation_history)

        # Generate response
        response = generate_response(
            prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )

        # Add assistant response to history
        conversation_history.append({"role": "assistant", "content": response})

        return ChatResponse(response=response, conversation=conversation_history)

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear_conversation():
    """Clear the conversation history."""
    global conversation_history
    conversation_history = []
    return JSONResponse(content={"success": True, "message": "Conversation cleared"})


@app.get("/api/status")
async def status():
    """Get current server status."""
    return JSONResponse(
        content={
            "model_loaded": model is not None,
            "current_checkpoint": (
                os.path.basename(current_checkpoint) if current_checkpoint else None
            ),
            "device": device,
            "conversation_length": len(conversation_history),
        }
    )


if __name__ == "__main__":
    # For direct execution only (development)
    # In production, use: uvicorn chat_ui.asgi:app --host 0.0.0.0 --port 8003
    import uvicorn

    # Get port from environment variable or default to 8003
    port = int(os.environ.get("CHAT_SERVER_PORT", 8003))

    print("🚀 Starting NanoGPT Chat Server...")
    print(f"📁 Checkpoint directory: {CHECKPOINT_DIR}")
    print(f"🌐 Server will be available at: http://0.0.0.0:{port}")
    print("💡 For production, use: uvicorn chat_ui.asgi:app --host 0.0.0.0 --port 8003")
    uvicorn.run(app, host="0.0.0.0", port=port, log_level="info")
