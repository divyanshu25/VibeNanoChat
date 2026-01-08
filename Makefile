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


.PHONY: uv uvlock venv dotenv environment jupyter-kernel kill-gpu gpu-hot gpu-status ddp-train chat


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


black-formatting:
	@echo "🔄 Formatting code with Black..."
	@$(uv) tool run black .
	@echo "✅ Code formatted with Black!"


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


ddp-train: ## Run DDP training. Usage: make ddp-train [NGPUS=2] [MODE=pretraining|mid-training|all] [CHECKPOINT=/path/to/checkpoint.pt] [NO_EVALS=true] [CORE_EVALS=true]
	@echo "🚀 Starting DDP training with torchrun..."
	@NGPUS=$${NGPUS:-2}; \
	MODE=$${MODE:-pretraining}; \
	CHECKPOINT=$${CHECKPOINT:-}; \
	NO_EVALS=$${NO_EVALS:-false}; \
	CORE_EVALS=$${CORE_EVALS:-false}; \
	echo "📊 Using $$NGPUS GPUs for distributed training"; \
	echo "🎯 Training mode: $$MODE"; \
	CMD="$(uv) run torchrun --standalone --nproc_per_node=$$NGPUS src/gpt_2/ddp.py --mode $$MODE"; \
	if [ -n "$$CHECKPOINT" ]; then \
		echo "📂 Using checkpoint: $$CHECKPOINT"; \
		CMD="$$CMD --checkpoint $$CHECKPOINT"; \
	fi; \
	if [ "$$NO_EVALS" = "true" ]; then \
		echo "⏩ Evaluations disabled"; \
		CMD="$$CMD --no-evals"; \
	fi; \
	if [ "$$CORE_EVALS" = "true" ]; then \
		echo "📊 CORE benchmark evaluations enabled"; \
		CMD="$$CMD --run-core-evals"; \
	fi; \
	echo ""; \
	eval $$CMD


chat: ## Chat with a checkpoint. Usage: make chat CHECKPOINT=/path/to/checkpoint.pt
	@if [ -z "$(CHECKPOINT)" ]; then \
		echo "❌ Error: CHECKPOINT is required. Usage: make chat CHECKPOINT=/path/to/checkpoint.pt"; \
		exit 1; \
	fi
	@echo "🤖 Starting chat with checkpoint: $(CHECKPOINT)"
	@$(uv) run python scripts/chat.py --checkpoint $(CHECKPOINT)

