---
title: Chat UI
tags: [chat, fastapi, server, inference, deployment]
---

# Chat UI

> [!ABSTRACT]
> Lightweight browser-based chat interface for interacting with trained checkpoints. FastAPI backend, Vanilla JS + Tailwind CSS frontend, no database required.

---

## Contents

- [[#Quick start]]
- [[#Configuration]]
- [[#API endpoints]]
- [[#Generation parameters]]
- [[#Architecture]]
- [[#Deployment]]

---

## Quick start

```bash
# Start the server (production mode: Gunicorn + Uvicorn workers)
make chat-server

# Development mode (auto-reload)
uv run uvicorn chat_ui.asgi:app --host 0.0.0.0 --port 8003 --reload
```

Open `http://localhost:8003`, select a checkpoint from the dropdown, click **Load Checkpoint**, and start chatting.

> [!NOTE] Checkpoint directory
> Before starting, set `CHECKPOINT_DIR` in `chat_ui/config.py` to point to your `.pt` checkpoint files.

---

## Configuration

### Checkpoint directory

`chat_ui/config.py`:
```python
CHECKPOINT_DIR = "<YOURPATH>/nanogpt/sft_checkpoints"
```

### Environment variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CHAT_SERVER_PORT` | 8003 | Port to bind to |
| `WORKERS` | 1 | Gunicorn workers (each loads the model independently) |
| `BIND_ADDRESS` | `0.0.0.0:8003` | Bind address |
| `LOG_LEVEL` | `info` | Logging verbosity |

```bash
# Example: localhost only, custom port
export CHAT_SERVER_PORT=8080
export BIND_ADDRESS="127.0.0.1:8080"
make chat-server
```

---

## API endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| `GET` | `/` | Serve the chat UI (static HTML) |
| `GET` | `/api/checkpoints` | List available checkpoints in `CHECKPOINT_DIR` |
| `POST` | `/api/load_checkpoint` | Load a specific checkpoint into memory |
| `POST` | `/api/chat` | Send a message and get a response |
| `POST` | `/api/clear` | Clear the current conversation history |
| `GET` | `/api/status` | Server status: model loaded, checkpoint name |

### `POST /api/chat`

Request:
```json
{
  "message": "What is the capital of France?",
  "temperature": 0.8,
  "top_k": 50,
  "max_tokens": 200
}
```

Response:
```json
{
  "response": "Paris is the capital of France...",
  "tokens_generated": 42
}
```

### `POST /api/load_checkpoint`

Request:
```json
{ "checkpoint": "step_19531.pt" }
```

---

## Generation parameters

| Parameter | Range | Default | Effect |
|-----------|-------|---------|--------|
| `temperature` | 0.1 – 2.0 | 0.8 | Lower = more focused; higher = more creative |
| `top_k` | 1 – 100 | 50 | Limits sampling to top K tokens |
| `max_tokens` | 50 – 500 | 200 | Maximum response length in tokens |

---

## Architecture

```
Browser
   │  HTTP
   ▼
FastAPI  (chat_ui/server.py)
   │
   ├── ModelManager
   │     — loads checkpoint on demand
   │     — holds model in GPU memory
   │     — runs autoregressive generation with KV cache
   │
   └── SessionManager
         — in-memory conversation history per session
         — cleared on server restart or POST /api/clear
```

> [!IMPORTANT] Constraints
> - Only **one checkpoint** can be loaded at a time
> - Conversation history is **not persisted** — resets on server restart
> - Multiple Gunicorn workers each load the model independently (no shared GPU memory)

---

## Deployment

### Option 1: Systemd service (recommended for production)

```bash
# Copy and configure the service file
sudo cp chat_ui/vibenanochat-chat.service /etc/systemd/system/

# Enable and start
sudo systemctl daemon-reload
sudo systemctl enable vibenanochat-chat
sudo systemctl start vibenanochat-chat

# Monitor
sudo systemctl status vibenanochat-chat
sudo journalctl -u vibenanochat-chat -f
```

### Option 2: Makefile (development)

```bash
make chat-server
```

Automatically clears port 8003 if already in use.

### Option 3: Direct Gunicorn

```bash
uv run gunicorn \
  --config chat_ui/gunicorn_config.py \
  chat_ui.asgi:application
```

---

## Related pages

- [[model]] — KV cache and generation parameters
- [[training]] — how checkpoints are saved
- [[architecture]] — inference data flow
