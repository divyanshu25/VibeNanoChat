#!/usr/bin/env python3
"""
FastAPI Chat Server for NanoGPT

Serves a web interface to chat with trained checkpoints.

Usage:
    python server.py
"""

import os
from pathlib import Path
from typing import List, Optional

from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field

from .config import ChatConfig
from .model_manager import ModelManager
from .prompt_utils import format_chat_prompt
from .session_manager import SessionManager

app = FastAPI(title="NanoGPT Chat Server")

# Initialize managers
model_manager = ModelManager()
session_manager = SessionManager()


# API Models
class ChatRequest(BaseModel):
    message: str
    session_id: Optional[str] = None
    max_tokens: Optional[int] = Field(
        default=ChatConfig.DEFAULT_MAX_TOKENS,
        ge=ChatConfig.MIN_TOKENS_LIMIT,
        le=ChatConfig.MAX_TOKENS_LIMIT,
    )
    temperature: Optional[float] = Field(
        default=ChatConfig.DEFAULT_TEMPERATURE,
        gt=ChatConfig.MIN_TEMPERATURE,
        le=ChatConfig.MAX_TEMPERATURE,
    )
    top_k: Optional[int] = Field(
        default=ChatConfig.DEFAULT_TOP_K,
        ge=ChatConfig.MIN_TOP_K,
        le=ChatConfig.MAX_TOP_K,
    )


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
        checkpoint_dir = Path(ChatConfig.CHECKPOINT_DIR)
        if not checkpoint_dir.exists():
            return JSONResponse(
                content={
                    "checkpoints": [],
                    "error": f"Directory not found: {ChatConfig.CHECKPOINT_DIR}",
                }
            )

        checkpoints = sorted(
            [str(f.relative_to(checkpoint_dir)) for f in checkpoint_dir.glob("**/*.pt")]
        )

        current = model_manager.get_current_checkpoint()
        current_basename = os.path.basename(current) if current else None

        return JSONResponse(
            content={
                "checkpoints": checkpoints,
                "current": current_basename,
            }
        )
    except Exception as e:
        return JSONResponse(content={"checkpoints": [], "error": str(e)})


@app.post("/api/load_checkpoint")
async def load_checkpoint_endpoint(request: CheckpointRequest):
    """Load a specific checkpoint."""
    try:
        # Resolve paths to prevent directory traversal attacks
        checkpoint_dir = Path(ChatConfig.CHECKPOINT_DIR).resolve()
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

        model_manager.load_model(str(requested_path))

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
    if not model_manager.is_loaded():
        raise HTTPException(
            status_code=400,
            detail="No checkpoint loaded. Please load a checkpoint first.",
        )

    try:
        # Get or create session
        session_id, conversation_history = session_manager.get_session(
            request.session_id
        )

        # Create temporary conversation with user message for prompt generation
        temp_conversation = conversation_history + [
            {"role": "user", "content": request.message}
        ]

        # Generate prompt from temporary conversation
        prompt = format_chat_prompt(temp_conversation)

        # Generate response
        response = model_manager.generate_response(
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
        session_manager.update_conversation(session_id, updated_conversation)

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

    success = session_manager.clear_conversation(session_id)
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
    session_id = session_manager.create_session()
    return SessionResponse(session_id=session_id, message="New session created")


@app.get("/api/session/{session_id}")
async def get_session_info(session_id: str):
    """Get information about a specific session."""
    if not session_manager.session_exists(session_id):
        raise HTTPException(status_code=404, detail="Session not found")

    session_data = session_manager.get_session_data(session_id)

    return JSONResponse(
        content={
            "session_id": session_data["session_id"],
            "conversation_length": session_data["conversation_length"],
            "last_accessed": session_data["last_accessed"].isoformat(),
            "conversation": session_data["conversation"],
        }
    )


@app.get("/api/status")
async def status():
    """Get current server status."""
    stats = session_manager.get_statistics()
    current_checkpoint = model_manager.get_current_checkpoint()

    return JSONResponse(
        content={
            "model_loaded": model_manager.is_loaded(),
            "current_checkpoint": (
                os.path.basename(current_checkpoint) if current_checkpoint else None
            ),
            "device": model_manager.get_device_name(),
            "active_sessions": stats["active_sessions"],
            "total_conversation_messages": stats["total_conversation_messages"],
        }
    )


if __name__ == "__main__":
    # For direct execution only (development)
    # In production, use: uvicorn chat_ui.server:app --host 0.0.0.0 --port 8003
    import uvicorn

    print("🚀 Starting NanoGPT Chat Server...")
    print(f"📁 Checkpoint directory: {ChatConfig.CHECKPOINT_DIR}")
    print(
        f"🌐 Server will be available at: http://{ChatConfig.HOST}:{ChatConfig.DEFAULT_PORT}"
    )
    print(
        "💡 For production, use: uvicorn chat_ui.server:app --host 0.0.0.0 --port 8003"
    )
    uvicorn.run(
        app, host=ChatConfig.HOST, port=ChatConfig.DEFAULT_PORT, log_level="info"
    )
