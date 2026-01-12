#!/usr/bin/env python3
"""
FastAPI Chat Server for NanoGPT

Serves a web interface to chat with trained checkpoints.

Usage:
    python server.py
"""

import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from threading import Lock
from typing import Dict, List, Optional

# Add gpt_2 to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.append(src_dir)

import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

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

# Session management
sessions: Dict[str, Dict] = (
    {}
)  # session_id -> {conversation: [], last_accessed: datetime}
sessions_lock = Lock()
SESSION_TIMEOUT_MINUTES = 60  # Sessions expire after 1 hour of inactivity


def get_device():
    """Determine the best available device."""
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def create_session() -> str:
    """Create a new session and return the session ID."""
    session_id = str(uuid.uuid4())
    with sessions_lock:
        sessions[session_id] = {
            "conversation": [],
            "last_accessed": datetime.now(),
        }
    return session_id


def get_session(session_id: Optional[str]) -> tuple[str, List[dict]]:
    """
    Get or create a session. Returns (session_id, conversation_history).

    If session_id is None or invalid, creates a new session.
    Updates last_accessed timestamp.
    """
    with sessions_lock:
        # Clean up expired sessions
        cleanup_expired_sessions()

        # If no session_id provided or session doesn't exist, create new one
        if not session_id or session_id not in sessions:
            session_id = str(uuid.uuid4())
            sessions[session_id] = {
                "conversation": [],
                "last_accessed": datetime.now(),
            }
        else:
            # Update last accessed time
            sessions[session_id]["last_accessed"] = datetime.now()

        return session_id, sessions[session_id]["conversation"]


def update_session_conversation(session_id: str, conversation: List[dict]):
    """Update the conversation history for a session."""
    with sessions_lock:
        if session_id in sessions:
            sessions[session_id]["conversation"] = conversation
            sessions[session_id]["last_accessed"] = datetime.now()


def clear_session(session_id: Optional[str]) -> bool:
    """Clear conversation history for a session. Returns True if session existed."""
    if not session_id:
        return False

    with sessions_lock:
        if session_id in sessions:
            sessions[session_id]["conversation"] = []
            sessions[session_id]["last_accessed"] = datetime.now()
            return True
    return False


def cleanup_expired_sessions():
    """Remove sessions that haven't been accessed in SESSION_TIMEOUT_MINUTES."""
    current_time = datetime.now()
    expired_sessions = [
        sid
        for sid, data in sessions.items()
        if (current_time - data["last_accessed"]).total_seconds()
        > SESSION_TIMEOUT_MINUTES * 60
    ]
    for sid in expired_sessions:
        del sessions[sid]

    if expired_sessions:
        print(f"🧹 Cleaned up {len(expired_sessions)} expired session(s)")


def load_model(checkpoint_path: str):
    """Load a checkpoint into the global model."""
    global model, tokenizer, device, current_checkpoint

    if device is None:
        device = get_device()
        print(f"🖥️  Device: {device}")

    if tokenizer is None:
        tokenizer, _ = get_custom_tokenizer()

    print(f"🔧 Loading checkpoint: {checkpoint_path}")

    # Create model as local variable first to avoid corrupting global state on failure
    new_model = None
    try:
        new_model = GPT(GPTConfig())
        new_model.to(device)
        load_checkpoint(
            checkpoint_path, new_model, device, optimizer=None, master_process=True
        )
        new_model.eval()

        # Only update global state after successful loading
        model = new_model
        current_checkpoint = checkpoint_path
        print(f"✅ Loaded: {os.path.basename(checkpoint_path)}")

    except Exception as e:
        # Ensure model remains None if loading fails
        model = None
        current_checkpoint = None
        print(f"❌ Failed to load checkpoint: {e}")
        raise  # Re-raise to propagate error to caller


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
    session_id: Optional[str] = None
    max_tokens: Optional[int] = Field(default=256, ge=1, le=1024)
    temperature: Optional[float] = Field(default=0.8, gt=0.0, le=2.0)
    top_k: Optional[int] = Field(default=50, ge=1, le=1000)


class ChatResponse(BaseModel):
    response: str
    conversation: List[dict]
    session_id: str


class CheckpointRequest(BaseModel):
    checkpoint: str


class SessionResponse(BaseModel):
    session_id: str
    message: str


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
        # Resolve paths to prevent directory traversal attacks
        checkpoint_dir = Path(CHECKPOINT_DIR).resolve()
        requested_path = (checkpoint_dir / request.checkpoint).resolve()

        # Validate that the resolved path is within the checkpoint directory
        try:
            requested_path.relative_to(checkpoint_dir)
        except ValueError:
            raise HTTPException(
                status_code=403, detail="Access denied: Path traversal detected"
            )

        # Check if file exists
        if not requested_path.exists():
            raise HTTPException(
                status_code=404, detail=f"Checkpoint not found: {request.checkpoint}"
            )

        # Verify it's a file (not a directory)
        if not requested_path.is_file():
            raise HTTPException(
                status_code=400, detail="Invalid checkpoint: Not a file"
            )

        load_model(str(requested_path))

        return JSONResponse(
            content={
                "success": True,
                "checkpoint": request.checkpoint,
                "message": f"Loaded {request.checkpoint}",
            }
        )
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/chat")
async def chat(request: ChatRequest):
    """Send a message and get a response."""
    if model is None:
        raise HTTPException(
            status_code=400,
            detail="No checkpoint loaded. Please load a checkpoint first.",
        )

    try:
        # Get or create session
        session_id, conversation_history = get_session(request.session_id)

        # Create temporary conversation with user message for prompt generation
        temp_conversation = conversation_history + [
            {"role": "user", "content": request.message}
        ]

        # Generate prompt from temporary conversation
        prompt = format_chat_prompt(temp_conversation)

        # Generate response
        response = generate_response(
            prompt,
            max_new_tokens=request.max_tokens,
            temperature=request.temperature,
            top_k=request.top_k,
        )

        # Only add to permanent history after successful generation
        updated_conversation = conversation_history + [
            {"role": "user", "content": request.message},
            {"role": "assistant", "content": response},
        ]
        update_session_conversation(session_id, updated_conversation)

        return ChatResponse(
            response=response, conversation=updated_conversation, session_id=session_id
        )

    except Exception as e:
        # Conversation history remains unchanged if generation fails
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/clear")
async def clear_conversation(session_id: Optional[str] = None):
    """Clear the conversation history for a session."""
    if not session_id:
        raise HTTPException(status_code=400, detail="session_id is required")

    success = clear_session(session_id)
    if not success:
        raise HTTPException(
            status_code=404, detail="Session not found or already expired"
        )

    return JSONResponse(
        content={
            "success": True,
            "message": "Conversation cleared",
            "session_id": session_id,
        }
    )


@app.post("/api/session/new")
async def new_session():
    """Create a new session."""
    session_id = create_session()
    return SessionResponse(session_id=session_id, message="New session created")


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session."""
    with sessions_lock:
        if session_id not in sessions:
            raise HTTPException(status_code=404, detail="Session not found")

        session_data = sessions[session_id]
        return JSONResponse(
            content={
                "session_id": session_id,
                "conversation_length": len(session_data["conversation"]),
                "last_accessed": session_data["last_accessed"].isoformat(),
                "conversation": session_data["conversation"],
            }
        )


@app.get("/api/status")
async def status():
    """Get current server status."""
    with sessions_lock:
        active_sessions = len(sessions)
        total_conversations = sum(len(s["conversation"]) for s in sessions.values())

    return JSONResponse(
        content={
            "model_loaded": model is not None,
            "current_checkpoint": (
                os.path.basename(current_checkpoint) if current_checkpoint else None
            ),
            "device": device,
            "active_sessions": active_sessions,
            "total_conversation_messages": total_conversations,
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
