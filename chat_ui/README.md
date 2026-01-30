# VibeNanoChat UI

A modern web interface for chatting with VibeNanoChat model checkpoints.

## Features

- 🎯 **Checkpoint Selection**: Browse and load any checkpoint from the SFT checkpoints directory
- 💬 **Interactive Chat**: Real-time conversation with the model
- 🎛️ **Advanced Controls**: Adjust temperature, top-k, and max tokens
- 🗑️ **Clear Conversation**: Reset chat history anytime
- 🎨 **Modern UI**: Built with Tailwind CSS for a clean, responsive design
- ⚡ **Fast**: No database required, all in-memory

## Requirements

Dependencies are managed through the project's `pyproject.toml` file. Install them using:

**Using uv (recommended)**:
```bash
cd /mnt/localssd/VibeNanoChat
uv sync
```

**Using pip**:
```bash
cd /mnt/localssd/VibeNanoChat
pip install -e .
```

This will install all required dependencies including FastAPI, Uvicorn, PyTorch, and tiktoken.

## Quick Start

### Start the Server
```bash
cd /mnt/localssd/VibeNanoChat
make chat-server
```

This will:
- Automatically clear port 8003 if it's in use
- Start gunicorn with uvicorn workers
- Use production-optimized settings
- Display the access URL

### Manual Options

**With gunicorn (recommended):**
```bash
cd /mnt/localssd/VibeNanoChat
uv run gunicorn --config chat_ui/gunicorn_config.py chat_ui.asgi:application
```

**With uvicorn (development with auto-reload):**
```bash
cd /mnt/localssd/VibeNanoChat
uv run uvicorn chat_ui.asgi:app --host 0.0.0.0 --port 8003 --reload
```

**Direct Python execution:**
```bash
cd /mnt/localssd/VibeNanoChat
uv run python chat_ui/server.py
```

### Accessing the UI

**Open your browser**:
Navigate to `http://localhost:8003` or `http://<your-server-ip>:8003`

1. **Select a checkpoint**:
Choose a checkpoint from the dropdown menu and click "Load Checkpoint"

2. **Start chatting**:
Type your message and press Enter or click Send

## Configuration

### Checkpoint Directory
The checkpoint directory is configured in `server.py`:
```python
CHECKPOINT_DIR = "/sensei-fs/users/divgoyal/nanogpt/sft_checkpoints"
```

You can change this to point to a different checkpoint directory if needed.

### Server Configuration

**Port Configuration:**
```bash
export CHAT_SERVER_PORT=8080
make chat-server
```

**Worker Configuration:**
```bash
# Use 2 workers (each loads the model independently)
export WORKERS=2
make chat-server
```

**Bind Address:**
```bash
# Localhost only (more secure)
export BIND_ADDRESS="127.0.0.1:8003"
make chat-server
```

**Log Level:**
```bash
export LOG_LEVEL=debug
make chat-server
```

## Advanced Settings

- **Temperature** (0.1 - 2.0): Controls randomness in generation
  - Lower values (0.3-0.7): More focused and deterministic
  - Higher values (0.9-1.5): More creative and random

- **Top-K** (1 - 100): Limits sampling to top K tokens
  - Lower values: More conservative
  - Higher values: More diverse

- **Max Tokens** (50 - 500): Maximum response length

## Keyboard Shortcuts

- `Enter`: Send message
- `Shift + Enter`: New line in input (not implemented, single line for now)

## Architecture

- **Backend**: FastAPI server with PyTorch model inference
- **ASGI Servers**: 
  - Development: Uvicorn with auto-reload
  - Production: Gunicorn with uvicorn workers
- **Frontend**: Vanilla JavaScript with Tailwind CSS
- **State Management**: In-memory conversation history (resets on server restart)

## Production Deployment

### Option 1: Systemd Service (Recommended)

1. Copy the service file:
```bash
sudo cp chat_ui/vibenanochat-chat.service /etc/systemd/system/
```

2. Edit the service file if needed:
```bash
sudo nano /etc/systemd/system/vibenanochat-chat.service
```

3. Enable and start the service:
```bash
sudo systemctl daemon-reload
sudo systemctl enable vibenanochat-chat
sudo systemctl start vibenanochat-chat
```

4. Check status:
```bash
sudo systemctl status vibenanochat-chat
```

5. View logs:
```bash
sudo journalctl -u vibenanochat-chat -f
```

### Option 2: Manual Run
```bash
cd /mnt/localssd/VibeNanoChat
make chat-server
```

## API Endpoints

- `GET /`: Serve the chat UI
- `GET /api/checkpoints`: List available checkpoints
- `POST /api/load_checkpoint`: Load a specific checkpoint
- `POST /api/chat`: Send a message and get response
- `POST /api/clear`: Clear conversation history
- `GET /api/status`: Get current server status

## Notes

- No database required - all conversation history is stored in memory
- Conversation history is cleared when the server restarts
- Only one checkpoint can be loaded at a time
- The server runs on port 8003 by default
- The Makefile target `make chat-server` automatically clears the port if it's in use
