"""
ASGI entry point for the NanoGPT Chat application.

This module provides the ASGI application interface for running the chat server
using uvicorn or other ASGI servers.

Usage with uvicorn (development):
    uvicorn chat_ui.asgi:app --host 0.0.0.0 --port 8003

Usage with gunicorn (production):
    gunicorn --config chat_ui/gunicorn_config.py chat_ui.asgi:application

Usage from project root:
    cd /mnt/localssd/NanoGPT
    uv run gunicorn --config chat_ui/gunicorn_config.py chat_ui.asgi:application
"""

import os

from chat_ui.server import app

# Get port from environment variable or default to 8003
PORT = int(os.environ.get("CHAT_SERVER_PORT", 8003))

# Expose the FastAPI app for ASGI servers (gunicorn uses 'application')
application = app

if __name__ == "__main__":
    # For development/testing only
    import uvicorn

    print("🚀 Starting NanoGPT Chat Server...")
    print(f"🌐 Server will be available at: http://0.0.0.0:{PORT}")
    uvicorn.run(app, host="0.0.0.0", port=PORT, log_level="info")
