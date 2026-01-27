"""
Gunicorn configuration for NanoGPT Chat Server.

This configuration provides production-ready settings for running the
chat server with optimal performance and reliability.

Documentation: https://docs.gunicorn.org/en/stable/settings.html
"""

import os

# ==============================================================================
# Server Socket
# ==============================================================================

# Server bind address
# WARNING: Binding to 0.0.0.0 makes the server accessible from all network interfaces
# For localhost-only access (more secure), use "127.0.0.1:8003"
# For production, ensure proper network segmentation, firewalls, and authentication
bind = os.getenv("BIND_ADDRESS", "0.0.0.0:8003")
backlog = 2048

# ==============================================================================
# Worker Processes
# ==============================================================================

# Number of worker processes
# For ML model serving, we use fewer workers to avoid memory issues
# Each worker will load the model into GPU/CPU memory
workers = int(os.getenv("WORKERS", "1"))  # Default to 1 worker for model serving

# Type of workers (for FastAPI/ASGI applications)
# 'uvicorn.workers.UvicornWorker' is the standard worker for FastAPI
worker_class = "uvicorn.workers.UvicornWorker"

# Note: threads parameter not applicable for uvicorn workers
# Uvicorn workers handle async I/O natively

# Maximum number of simultaneous clients
worker_connections = 1000

# Maximum requests a worker will process before restarting
# Helps prevent memory leaks from model inference
max_requests = 1000
max_requests_jitter = 100

# Workers silent for more than this many seconds are killed and restarted
# Increased timeout for model inference operations
timeout = 300  # 5 minutes for long inference operations

# Graceful timeout for worker shutdown
graceful_timeout = 60

# Keep alive timeout
keepalive = 5

# ==============================================================================
# Logging
# ==============================================================================

# Access log
accesslog = "-"  # Log to stdout
access_log_format = '%(h)s %(l)s %(u)s %(t)s "%(r)s" %(s)s %(b)s "%(f)s" "%(a)s" %(D)s'

# Error log
errorlog = "-"  # Log to stderr
loglevel = os.getenv("LOG_LEVEL", "info").lower()

# Capture stdout/stderr to log
capture_output = True

# ==============================================================================
# Process Naming
# ==============================================================================

proc_name = "nanogpt-chat"

# ==============================================================================
# Server Mechanics
# ==============================================================================

# Daemonize the Gunicorn process (detach from terminal)
daemon = False

# A directory to store temporary request data
tmp_upload_dir = None

# ==============================================================================
# Preload Application
# ==============================================================================

# Preload application code before worker processes are forked
# WARNING: For ML models, set to False to avoid memory issues
# Each worker should load the model independently
preload_app = False

# ==============================================================================
# SSL (if needed)
# ==============================================================================

# keyfile = "/path/to/keyfile"
# certfile = "/path/to/certfile"

# ==============================================================================
# Server Hooks
# ==============================================================================


def on_starting(server):
    """Called just before the master process is initialized."""
    print("🚀 NanoGPT Chat Server (FastAPI) starting...")
    print(f"📍 Server binding to: {bind}")
    print(f"👷 Workers: {workers} (uvicorn async workers)")
    print(f"⏱️  Timeout: {timeout}s (for model inference)")
    print("📁 Checkpoint directory: /sensei-fs/users/divgoyal/nanogpt/sft_checkpoints")
    print("🌐 Access URLs:")
    print("   - Chat UI:      http://localhost:8003/")
    print("   - API Docs:     http://localhost:8003/docs")
    print("   - OpenAPI:      http://localhost:8003/openapi.json")


def on_reload(server):
    """Called to recycle workers during a reload via SIGHUP."""
    print("🔄 Reloading workers...")


def when_ready(server):
    """Called just after the server is started."""
    print("✅ Server is ready to accept connections!")
    print("💡 Load a checkpoint from the UI to start chatting")


def worker_int(worker):
    """Called when a worker receives the INT or QUIT signal."""
    print(f"👷 Worker {worker.pid} interrupted")


def worker_abort(worker):
    """Called when a worker receives the SIGABRT signal."""
    print(f"⚠️  Worker {worker.pid} aborted")


def pre_fork(server, worker):
    """Called just before a worker is forked."""
    pass


def post_fork(server, worker):
    """Called just after a worker has been forked."""
    pass


def post_worker_init(worker):
    """Called just after a worker has initialized the application."""
    print(f"👷 Worker {worker.pid} initialized and ready for model loading")


def worker_exit(server, worker):
    """Called just after a worker has been exited."""
    print(f"👷 Worker {worker.pid} exited")


def nworkers_changed(server, new_value, old_value):
    """Called when the number of workers changes."""
    print(f"👷 Workers changed: {old_value} → {new_value}")


def on_exit(server):
    """Called just before exiting Gunicorn."""
    print("👋 Server shutting down...")
