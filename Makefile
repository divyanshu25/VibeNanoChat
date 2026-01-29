# Use bash as the default shell
SHELL := /bin/bash

# Service name
SERVICE_NAME := nano-gpt

# Define Python version
PYTHON_VERSION := 3.10

# Define the virtual environment directory.
ENV_TARGET_DIR := .



# Define log level
export LOG_LEVEL ?= INFO

# Define the virtual environment uv path.
uv := $(HOME)/.local/bin/uv
ifneq ($(shell which uv),)
	override uv := $(shell which uv)
endif


.PHONY: help uv uvlock venv dotenv environment jupyter-kernel format lint check kill-gpu gpu-hot gpu-status ddp-train run-scaling-law chat-server

.DEFAULT_GOAL := help

help: ## Show this help message
	@echo "📚 Available targets:"
	@echo ""
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | awk 'BEGIN {FS = ":.*?## "}; {printf "  \033[36m%-20s\033[0m %s\n", $$1, $$2}'
	@echo ""
	@echo "💡 Usage: make <target>"


dotenv: ## Initialize .env file
	@echo "📝 Creating .env file from template..."
	@cp -n .env.template $(ENV_TARGET_DIR)/.env || echo "⚠️  $(ENV_TARGET_DIR)/.env already exists. Skipping copy."


uv:  ## INSTALL UV
ifeq ($(shell PATH=$(PATH) which uv),)
ifneq ($(shell which brew),) #macos
	@echo 
	@echo "Installing UV with Homebrew"
	@brew install uv
	$(eval override uv := $(shell brew --prefix)/bin/uv)
else
	@echo
	@echo "⬇️  Installing UV with a script..."
	@curl -LsSf https://astral.sh/uv/install.sh | sh
	@echo
endif
endif
	@echo "✅ uv is already installed at $(uv)"
	@echo


uvlock: ## Sync project with uv
	@echo "🔄 Syncing project dependencies with uv..."
	@if [ ! -f "uv.lock" ]; then \
		echo "🆕 uv.lock file not found. Creating a new one..."; \
		$(uv) lock; \
	fi
	@echo "✅ UV lock file is ready!"

venv: dotenv ## Create virtual environment
	@echo "🐍 Setting up your Python virtual environment..."
	@$(uv) tool run --from 'python-dotenv[cli]' dotenv run $(uv) venv --python $(PYTHON_VERSION)
	@$(uv) tool run --from 'python-dotenv[cli]' dotenv run $(uv) sync --frozen
	@echo "🎉 Virtual environment setup complete!"

environment: uv uvlock venv ## Create environment
	@echo "🚀 All set! Your environment is ready."
	@echo
	@echo "💡 Quick start commands:"
	@echo "   👉  To activate: source .venv/bin/activate"
	@echo "✨ Happy coding with NanoGPT!"


jupyter-kernel: venv ## Register environment as Jupyter kernel
	@echo "📝 Registering Jupyter kernel..."
	@$(uv) pip install ipykernel
	@$(uv) run python -m ipykernel install --user --name=nanogpt --display-name="NanoGPT"
	@echo "✅ Jupyter kernel 'NanoGPT' registered!"
	@echo "💡 Select 'NanoGPT' as your kernel in Jupyter notebooks"
	@echo


format: ## Format code with Black and isort
	@echo "🎨 Formatting code..."
	@$(uv) tool run black .
	@$(uv) tool run isort .
	@echo "✅ Code formatted!"

lint: ## Run linting with ruff
	@echo "🔍 Linting with ruff..."
	@$(uv) tool run ruff check .
	@echo "✅ Linting complete!"

check: format lint ## Run format + lint
	@echo "✅ All checks passed!"


kill-gpu: ## Kill all GPU processes
	@echo "🔪 Killing all GPU processes..."
	@nvidia-smi --query-compute-apps=pid --format=csv,noheader | xargs -r kill -9
	@echo "✅ All GPU processes stopped!"
	@echo "💡 Run 'nvidia-smi' to verify GPUs are free"


gpu-hot: ## Keep GPUs at ~90%+ utilization. Usage: make gpu-hot [GPUS=0,1,2] [DELAY=2]
	@echo "🔥 Starting GPU heating..."
	@DELAY_ARG=""; \
	if [ -n "$(DELAY)" ]; then \
		DELAY_ARG="--delay $(DELAY)"; \
		echo "⏳ Will wait $(DELAY) hour(s) before starting"; \
	fi; \
	if [ -z "$(GPUS)" ]; then \
		$(uv) run python scripts/keep_gpus_hot.py $$DELAY_ARG; \
	else \
		$(uv) run python scripts/keep_gpus_hot.py $(GPUS) $$DELAY_ARG; \
	fi


gpu-status: ## Show current GPU utilization and memory usage
	@nvidia-smi

# Sample Pretrain: make ddp-train NGPUS=2 MODE=pretraining CORE_EVALS=true DEPTH=12 TARGET_FLOPS=1e18
# Sample Midtrain: make ddp-train NGPUS=2 MODE=mid-training CHATCORE_EVALS=true DEPTH=12 CHECKPOINT=/sensei-fs/users/divgoyal/nanogpt/pretrain_checkpoints/model_checkpoint_global37953_pretraining.pt
# Sample SFT: make ddp-train NGPUS=2 MODE=sft CHATCORE_EVALS=true DEPTH=12 CHECKPOINT=/sensei-fs/users/divgoyal/nanogpt/pretrain_checkpoints/model_checkpoint_global37953_pretraining.pt
ddp-train: ## Run DDP training. Usage: make ddp-train [NGPUS=2] [MODE=pretraining|mid-training|all] [CHECKPOINT=/path/to/checkpoint.pt] [VAL_EVALS=true] [CORE_EVALS=true] [CHATCORE_EVALS=true] [DEPTH=12] [TARGET_FLOPS=1e18] [EVAL_INTERVAL=500]
	@echo "🚀 Starting DDP training with torchrun..."
	@mkdir -p logs
	@NGPUS=$${NGPUS:-4}; \
	MODE=$${MODE:-pretraining}; \
	CHECKPOINT=$${CHECKPOINT:-}; \
	VAL_EVALS=$${VAL_EVALS:-true}; \
	CORE_EVALS=$${CORE_EVALS:-false}; \
	CHATCORE_EVALS=$${CHATCORE_EVALS:-false}; \
	DEPTH=$${DEPTH:-}; \
	ASPECT_RATIO=$${ASPECT_RATIO:-64}; \
	HEAD_DIM=$${HEAD_DIM:-128}; \
	TARGET_FLOPS=$${TARGET_FLOPS:-}; \
	EVAL_INTERVAL=$${EVAL_INTERVAL:-}; \
	TIMESTAMP=$$(date +%Y%m%d_%H%M%S); \
	LOG_FILE="logs/scaling_laws_N$${DEPTH}_F$${TARGET_FLOPS}.log"; \
	echo "📊 Using $$NGPUS GPUs for distributed training"; \
	echo "🎯 Training mode: $$MODE"; \
	echo "📝 Logging to: $$LOG_FILE"; \
	CMD="$(uv) run torchrun --standalone --nproc_per_node=$$NGPUS src/gpt_2/ddp.py --mode $$MODE"; \
	if [ -n "$$CHECKPOINT" ]; then \
		echo "📂 Using checkpoint: $$CHECKPOINT"; \
		CMD="$$CMD --checkpoint $$CHECKPOINT"; \
	fi; \
	if [ "$$VAL_EVALS" = "false" ]; then \
		echo "⏩ Validation evaluations disabled"; \
		CMD="$$CMD --no-evals"; \
	else \
		echo "✅ Validation evaluations enabled"; \
	fi; \
	if [ "$$CORE_EVALS" = "true" ]; then \
		echo "📊 CORE benchmark evaluations enabled"; \
		CMD="$$CMD --run-core-evals"; \
	fi; \
	if [ "$$CHATCORE_EVALS" = "true" ]; then \
		echo "💬 ChatCore evaluations enabled"; \
		CMD="$$CMD --run-chatcore-evals"; \
	fi; \
	if [ -n "$$DEPTH" ]; then \
		echo "📐 Using depth-based architecture: depth=$$DEPTH (aspect_ratio=$$ASPECT_RATIO, head_dim=$$HEAD_DIM)"; \
		CMD="$$CMD --depth $$DEPTH --aspect-ratio $$ASPECT_RATIO --head-dim $$HEAD_DIM"; \
	fi; \
	if [ -n "$$TARGET_FLOPS" ]; then \
		echo "🎯 Target FLOPs: $$TARGET_FLOPS"; \
		CMD="$$CMD --target-flops $$TARGET_FLOPS"; \
	fi; \
	if [ -n "$$EVAL_INTERVAL" ]; then \
		echo "⏱️  Eval interval: $$EVAL_INTERVAL steps"; \
		CMD="$$CMD --eval-interval $$EVAL_INTERVAL"; \
	fi; \
	echo ""; \
	eval $$CMD 2>&1 | tee $$LOG_FILE


run-scaling-law: ## Run scaling law experiment with nanochat-style depth and FLOP budget sweep
	@echo "🔬 Starting scaling law experiments (depth × FLOP budget sweep)..."
	@echo "📊 Using adaptive eval_interval (~4 evals per run, scales with model size)"
	@for FLOPS in 1e18 3e18 6e18; do \
		echo ""; \
		echo "================================================================="; \
		echo "💰 Compute budget: $$FLOPS FLOPs"; \
		echo "================================================================="; \
		for DEPTH in 6 8 10 12 14 16 18; do \
			echo ""; \
			echo "  🧪 depth=$$DEPTH at $$FLOPS FLOPs"; \
			$(MAKE) ddp-train NGPUS=4 MODE=pretraining CORE_EVALS=true DEPTH=$$DEPTH TARGET_FLOPS=$$FLOPS EVAL_INTERVAL=100|| exit 1; \
			echo "  🧹 Cleaning up GPUs..."; \
			$(MAKE) kill-gpu; \
			sleep 20; \
			echo "  🧹 Cleaning up GPUs..."; \
			$(MAKE) kill-gpu; \
			sleep 20; \
		done; \
	done
	@echo "✅ All scaling law experiments complete!"

chat-server: ## Start the chat web UI server on port 8003
	@echo "🔍 Checking if port 8003 is in use..."
	@PORT_PID=$$(lsof -ti:8003); \
	if [ -n "$$PORT_PID" ]; then \
		echo "⚠️  Port 8003 is in use by PID $$PORT_PID. Killing process..."; \
		kill -9 $$PORT_PID 2>/dev/null || true; \
		sleep 1; \
		echo "✅ Port 8003 cleared!"; \
	else \
		echo "✅ Port 8003 is available"; \
	fi
	@echo "🚀 Starting NanoGPT Chat Web Server"
	@echo "🌐 Access the UI at: http://localhost:8003"
	@echo "📁 Checkpoint directory: /sensei-fs/users/divgoyal/nanogpt/midtrain_checkpoints"
	@echo "👷 Workers: $${WORKERS:-1} (set WORKERS env var to change)"
	@$(uv) run gunicorn --config chat_ui/gunicorn_config.py chat_ui.asgi:application

