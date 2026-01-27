"""
Functions for evaluating the CORE metric, as described in the DCLM paper.
https://arxiv.org/abs/2406.11794

This module provides functionality for evaluating language models on various task types:
- Multiple Choice: Select best completion from choices (e.g., HellaSwag, ARC)
- Language Modeling: Exact token prediction (e.g., LAMBADA, SQuAD)
- Schema: Match context options to continuation (e.g., Winograd)

TODOs:
- All tasks ~match except for squad. We get 31% reference is 37%. Figure out why.
"""

import random
from typing import Any, Dict, List, Optional, Tuple

import torch
import torch.distributed as dist

from .prompt_rendering import (render_prompts_lm, render_prompts_mc,
                               render_prompts_schema)
from .token_batching import (batch_sequences_lm, batch_sequences_mc,
                             batch_sequences_schema, stack_sequences)

# =============================================================================
# MODEL EVALUATION
# =============================================================================


@torch.no_grad()
def forward_model(model, input_ids: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Forward pass through model to compute per-token losses and predictions.

    Args:
        model: Language model to evaluate
        input_ids: Tensor of shape (batch_size, seq_len) with token IDs

    Returns:
        losses: Tensor of shape (batch_size, seq_len) with cross-entropy loss at each position.
                Last column is NaN since there's no autoregressive target for it.
        predictions: Tensor of shape (batch_size, seq_len) with argmax predictions at each position
    """
    batch_size, seq_len = input_ids.size()
    model_output = model(input_ids)

    # Handle both single tensor output and tuple output (logits, loss)
    if isinstance(model_output, tuple):
        outputs = model_output[0]  # Get logits from (logits, loss) tuple
    else:
        outputs = model_output

    # Shift input_ids left by one to get autoregressive targets
    # Example: input_ids = [The, cat, sat, on, mat]
    #          target_ids = [cat, sat, on, mat, ?]
    # This is because in language modeling, predictions[i] predicts input_ids[i+1]
    target_ids = torch.roll(
        input_ids, shifts=-1, dims=1
    )  # this will shift the input_ids to the left by one position

    # Calculate cross entropy at all positions
    # losses[i] = how well the model predicted input_ids[i+1] from position i
    losses = torch.nn.functional.cross_entropy(
        outputs.view(batch_size * seq_len, -1),
        target_ids.view(batch_size * seq_len),
        reduction="none",
    ).view(batch_size, seq_len)

    # Set the last column to NaN (no autoregressive target for final token)
    losses[:, -1] = float("nan")

    # Get argmax predictions at each position
    # predictions[i] = model's predicted token for position i+1 (autoregressive)
    predictions = outputs.argmax(dim=-1)
    return losses, predictions


@torch.no_grad()
def evaluate_example(
    idx: int,
    model,
    tokenizer,
    data: List[Dict[str, Any]],
    device,
    task_meta: Dict[str, Any],
) -> Optional[bool]:
    """
    Evaluate a single example from the dataset.

    Process:
    1. Sample few-shot examples for in-context learning
    2. Render prompts based on task type
    3. Tokenize and prepare sequences
    4. Truncate if needed to fit model's max sequence length
    5. Forward through model and compute correctness

    Args:
        idx: Index of the example in the data list
        model: Language model to evaluate
        tokenizer: Tokenizer instance
        data: List of all evaluation examples
        device: Device to run evaluation on
        task_meta: Dict with 'task_type', 'num_fewshot', 'continuation_delimiter'

    Returns:
        True if prediction is correct, False if incorrect, None if example can't be evaluated
        (e.g., sequence too long even after truncation)
    """
    item = data[idx]
    task_type = task_meta["task_type"]
    num_fewshot = task_meta["num_fewshot"]
    continuation_delimiter = task_meta["continuation_delimiter"]

    # -------------------------------------------------------------------------
    # Step 1: Sample few-shot examples (excluding current item)
    # -------------------------------------------------------------------------
    fewshot_examples = []
    if num_fewshot > 0:
        rng = random.Random(1234 + idx)  # Deterministic sampling
        available_indices = [i for i in range(len(data)) if i != idx]
        fewshot_indices = rng.sample(
            available_indices, min(num_fewshot, len(available_indices))
        )
        fewshot_examples = [data[i] for i in fewshot_indices]

    # -------------------------------------------------------------------------
    # Step 2: Render prompts and tokenize based on task type
    # -------------------------------------------------------------------------
    if task_type == "multiple_choice":
        # Example: HellaSwag - "A man is sitting on a roof. He"
        # Creates 4 prompts (one per choice):
        #   - "A man is sitting on a roof. He starts pulling up roofing..."
        #   - "A man is sitting on a roof. He is using a laptop..."
        #   - etc.
        # Returns: tokens for all 4, start_idxs=[10,10,10,10] (where choices start),
        #          end_idxs=[17,17,14,16] (length of each sequence)
        prompts = render_prompts_mc(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_mc(tokenizer, prompts)

    elif task_type == "schema":
        # Example: Winograd - Different contexts, same continuation
        #   - "The trophy doesn't fit because it is too large. The trophy is too big."
        #   - "The trophy doesn't fit because it is too small. The trophy is too big."
        # Returns: tokens for all contexts, start_idxs=[17,17] (where continuation starts),
        #          end_idxs=[22,22] (length of each sequence)
        prompts = render_prompts_schema(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_schema(tokenizer, prompts)

    elif task_type == "language_modeling":
        # Example: "The capital of France is" → " Paris"
        # Creates 2 prompts: without and with continuation
        #   - "The capital of France is"
        #   - "The capital of France is Paris"
        # Returns: tokens for WITH continuation only, start_idxs=[6] (where " Paris" starts),
        #          end_idxs=[7] (total length)
        prompts = render_prompts_lm(item, continuation_delimiter, fewshot_examples)
        tokens, start_idxs, end_idxs = batch_sequences_lm(tokenizer, prompts)

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    # -------------------------------------------------------------------------
    # Step 3: Truncate sequences if they exceed model's max length
    # -------------------------------------------------------------------------
    # Example: If model.max_seq_len = 1024 and a sequence is 1200 tokens:
    #   Original: tokens=[0...1199], start_idx=1150, end_idx=1200
    #   After:    tokens=[176...1199] (last 1024 tokens), start_idx=974, end_idx=1024
    #   This keeps the continuation intact (prioritizes the end)
    if hasattr(model, "max_seq_len") and model.max_seq_len is not None:
        max_tokens = model.max_seq_len
        new_tokens, new_start_idxs, new_end_idxs = [], [], []

        for t, s, e in zip(tokens, start_idxs, end_idxs):
            if len(t) > max_tokens:
                num_to_crop = (
                    len(t) - max_tokens
                )  # e.g., 1200 - 1024 = 176 tokens to remove
                # Keep the last max_tokens (prioritize the continuation)
                new_tokens.append(t[-max_tokens:])  # Keep tokens [176:1200]
                new_start_idxs.append(s - num_to_crop)  # Adjust: 1150 - 176 = 974
                new_end_idxs.append(e - num_to_crop)  # Adjust: 1200 - 176 = 1024

                # If truncation cuts into the evaluation region, skip this example
                # (continuation would be incomplete, can't fairly evaluate)
                if s - num_to_crop < 0 or e - num_to_crop < 0:
                    return None  # Can't evaluate this example
            else:
                # No truncation needed
                new_tokens.append(t)
                new_start_idxs.append(s)
                new_end_idxs.append(e)

        tokens, start_idxs, end_idxs = new_tokens, new_start_idxs, new_end_idxs

    # -------------------------------------------------------------------------
    # Step 4: Create batch tensor and run model forward pass
    # -------------------------------------------------------------------------
    pad_token_id = tokenizer.get_bos_token_id()  # BOS token works fine as padding
    input_ids = stack_sequences(tokens, pad_token_id)
    input_ids = input_ids.to(device)
    losses, predictions = forward_model(model, input_ids)

    # -------------------------------------------------------------------------
    # Step 5: Compute correctness based on task type
    # -------------------------------------------------------------------------
    if task_type == "language_modeling":
        # Check if all tokens in continuation are predicted exactly
        #
        # AUTOREGRESSIVE: predictions[i] predicts input_ids[i+1]
        # Example: input_ids = [BOS, The, capital, of, France, is, Paris]
        #                        0    1     2      3    4      5    6
        #          predictions[5] predicts input_ids[6] which should be "Paris"
        #
        # If start_idx=6, end_idx=7 (the continuation " Paris"):
        #   predicted_tokens = predictions[0, 5:6] = what model predicted for position 6
        #   actual_tokens = input_ids[0, 6:7] = "Paris" (the actual token)
        #   is_correct = True if predicted_tokens == actual_tokens
        si = start_idxs[0]
        ei = end_idxs[0]
        predicted_tokens = predictions[0, si - 1 : ei - 1]  # What model predicted
        actual_tokens = input_ids[0, si:ei]  # Ground truth
        is_correct = torch.all(predicted_tokens == actual_tokens).item()

    elif task_type in ["multiple_choice", "schema"]:
        # Select the option with lowest average loss over the continuation
        # Lower loss = model thinks this continuation is more likely (better fit)
        #
        # AUTOREGRESSIVE: losses[i] = loss for predicting input_ids[i+1] at position i
        # We use si-1:ei-1 because loss is offset by 1 from the actual tokens
        #
        # Example: Multiple choice with 4 options
        #   losses[0, si:ei] = [0.5, 0.3, 0.4] → mean = 0.4  (option A)
        #   losses[1, si:ei] = [0.2, 0.1, 0.2] → mean = 0.167 (option B) ← lowest!
        #   losses[2, si:ei] = [1.2, 0.9, 1.1] → mean = 1.067 (option C)
        #   losses[3, si:ei] = [0.8, 0.7, 0.6] → mean = 0.7  (option D)
        #   pred_idx = 1 (option B has lowest loss)
        #   is_correct = True if item["gold"] == 1
        mean_losses = [
            losses[i, si - 1 : ei - 1].mean().item()
            for i, (si, ei) in enumerate(zip(start_idxs, end_idxs))
        ]
        pred_idx = mean_losses.index(min(mean_losses))  # Pick option with lowest loss
        is_correct = pred_idx == item["gold"]  # Compare to correct answer

    else:
        raise ValueError(f"Unsupported task type: {task_type}")

    return is_correct


def evaluate_task(
    model,
    tokenizer,
    data: List[Dict[str, Any]],
    device,
    task_meta: Dict[str, Any],
    max_examples: Optional[int] = None,
) -> float:
    """
    Evaluate a model on a complete task across many examples.

    Supports distributed evaluation: automatically stripes examples across
    multiple GPUs/processes if run with torchrun, then aggregates results.

    Args:
        model: The language model to evaluate
        tokenizer: Tokenizer instance
        data: List of evaluation examples for this task
        device: Device to run evaluation on
        task_meta: Task metadata dict with 'task_type', 'num_fewshot', 'continuation_delimiter'
        max_examples: Optional limit on number of examples (useful for faster iteration during training)

    Returns:
        Mean accuracy across all successfully evaluated examples (0.0 to 1.0)
    """
    # Get distributed training info (if applicable)
    rank = dist.get_rank() if dist.is_initialized() else 0
    world_size = dist.get_world_size() if dist.is_initialized() else 1

    # Optionally limit number of examples for faster evaluation
    num_examples = min(len(data), max_examples) if max_examples else len(data)

    # Track correctness and which examples were successfully evaluated
    correct = torch.zeros(num_examples, dtype=torch.float32, device=device)
    evaluated = torch.zeros(num_examples, dtype=torch.float32, device=device)

    # Distribute examples across ranks (each rank handles every world_size-th example)
    for idx in range(rank, num_examples, world_size):
        is_correct = evaluate_example(idx, model, tokenizer, data, device, task_meta)
        if is_correct is not None:  # Some examples may be skipped if too long
            correct[idx] = float(is_correct)
            evaluated[idx] = 1.0

    # Aggregate results across all processes if running distributed
    if world_size > 1:
        dist.barrier()
        dist.all_reduce(correct, op=dist.ReduceOp.SUM)
        dist.all_reduce(evaluated, op=dist.ReduceOp.SUM)

    # Compute mean accuracy over successfully evaluated examples
    total_evaluated = evaluated.sum().item()
    if total_evaluated == 0:
        return 0.0
    mean_correct = correct.sum().item() / total_evaluated
    return mean_correct
