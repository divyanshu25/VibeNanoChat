"""
Model management for the NanoGPT Chat Server.

Handles model loading, device management, and text generation.
"""

import os
import sys
from typing import List, Optional

# Add gpt_2 to python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
src_dir = os.path.join(parent_dir, "src")
sys.path.append(src_dir)

import torch
import torch.nn.functional as F

from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT
from gpt_2.utils import get_custom_tokenizer, load_checkpoint

from .config import ChatConfig


class ModelManager:
    """Manages model loading and inference for the chat server."""

    def __init__(self):
        """Initialize the model manager."""
        self.model: Optional[GPT] = None
        self.tokenizer = None
        self.device: Optional[str] = None
        self.current_checkpoint: Optional[str] = None
        self._stop_tokens: Optional[List[int]] = None

    def get_device(self) -> str:
        """Determine the best available device."""
        if torch.cuda.is_available():
            return "cuda"
        elif torch.backends.mps.is_available():
            return "mps"
        return "cpu"

    def load_model(self, checkpoint_path: str):
        """
        Load a checkpoint into the model.

        Args:
            checkpoint_path: Path to the checkpoint file.

        Raises:
            Exception: If the checkpoint cannot be loaded.
        """
        if self.device is None:
            self.device = self.get_device()
            print(f"🖥️  Device: {self.device}")

        if self.tokenizer is None:
            self.tokenizer, _ = get_custom_tokenizer()
            self._initialize_stop_tokens()

        print(f"🔧 Loading checkpoint: {checkpoint_path}")

        # Create model as local variable first to avoid corrupting state on failure
        new_model = None
        try:
            new_model = GPT(GPTConfig())
            new_model.to(self.device)
            load_checkpoint(
                checkpoint_path,
                new_model,
                self.device,
                optimizer=None,
                master_process=True,
            )
            new_model.eval()

            # Only update state after successful loading
            self.model = new_model
            self.current_checkpoint = checkpoint_path
            print(f"✅ Loaded: {os.path.basename(checkpoint_path)}")

        except Exception as e:
            # Ensure model remains None if loading fails
            self.model = None
            self.current_checkpoint = None
            print(f"❌ Failed to load checkpoint: {e}")
            raise  # Re-raise to propagate error to caller

    def _initialize_stop_tokens(self):
        """Initialize the list of stop tokens for generation."""
        if self.tokenizer is None:
            return

        self._stop_tokens = [
            self.tokenizer.encode(
                ChatConfig.CHAT_TOKENS["assistant_end"], allowed_special="all"
            )[0],
            self.tokenizer.encode(
                ChatConfig.CHAT_TOKENS["endoftext"], allowed_special="all"
            )[0],
            self.tokenizer.encode(
                ChatConfig.CHAT_TOKENS["user_start"], allowed_special="all"
            )[0],
            self.tokenizer.encode(
                ChatConfig.CHAT_TOKENS["user_end"], allowed_special="all"
            )[0],
        ]

    def generate_response(
        self,
        prompt: str,
        max_new_tokens: int = ChatConfig.DEFAULT_MAX_TOKENS,
        temperature: float = ChatConfig.DEFAULT_TEMPERATURE,
        top_k: int = ChatConfig.DEFAULT_TOP_K,
    ) -> str:
        """
        Generate a response from the model given a prompt.

        Args:
            prompt: The input prompt.
            max_new_tokens: Maximum number of tokens to generate.
            temperature: Sampling temperature.
            top_k: Top-k sampling parameter.

        Returns:
            The generated response text.

        Raises:
            ValueError: If no model is loaded.
        """
        if self.model is None:
            raise ValueError("No model loaded")

        if self.tokenizer is None:
            raise ValueError("No tokenizer loaded")

        self.model.eval()

        tokens = self.tokenizer.encode(prompt, allowed_special="all")
        tokens = torch.tensor(tokens, dtype=torch.long, device=self.device).unsqueeze(0)
        prompt_length = tokens.size(1)

        with torch.no_grad():
            for _ in range(max_new_tokens):
                if tokens.size(1) > self.model.config.block_size:
                    tokens = tokens[:, -self.model.config.block_size :]

                logits, _ = self.model(tokens)
                logits = logits[:, -1, :] / temperature

                # Top-k sampling
                v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                logits[logits < v[:, [-1]]] = float("-inf")

                probs = F.softmax(logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                tokens = torch.cat([tokens, next_token], dim=1)

                if next_token.item() in self._stop_tokens:
                    break

        generated_tokens = tokens[0, prompt_length:].tolist()
        raw_response = self.tokenizer.decode(generated_tokens)

        # Clean up special tokens
        for token_str in ChatConfig.CHAT_TOKENS.values():
            raw_response = raw_response.replace(token_str, "")

        return raw_response.strip()

    def is_loaded(self) -> bool:
        """Check if a model is currently loaded."""
        return self.model is not None

    def get_current_checkpoint(self) -> Optional[str]:
        """Get the name of the currently loaded checkpoint."""
        return self.current_checkpoint

    def get_device_name(self) -> Optional[str]:
        """Get the name of the device being used."""
        return self.device
