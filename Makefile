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


.PHONY: uv uvlock venv dotenv environment jupyter-kernel


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







