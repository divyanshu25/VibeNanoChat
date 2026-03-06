"""Evaluator setup utilities for the trainer."""

from eval_tasks import CoreEvaluator
from eval_tasks.training import TrainingEvaluator
from gpt_2.utils import get_custom_tokenizer


def setup_chatcore_evaluator(
    raw_model,
    device,
    master_process,
    ddp,
    ddp_rank,
    ddp_world_size,
    config,
):
    """
    Setup the ChatCORE evaluator for chat-based benchmark tasks.

    Args:
        raw_model: The raw (unwrapped) model
        device: Device to use
        master_process: Whether this is the master process
        ddp: Whether using DDP
        ddp_rank: DDP rank
        ddp_world_size: DDP world size
        config: GPTConfig instance

    Returns:
        ChatCoreEvaluator: The ChatCORE evaluator instance with registered tasks
    """
    from eval_tasks.chat_core.arc_challenge import setup_arc_challenge_task
    from eval_tasks.chat_core.arc_easy import setup_arc_task
    from eval_tasks.chat_core.evaluator import ChatCoreEvaluator
    from eval_tasks.chat_core.gsm8k import setup_gsm8k_task
    from eval_tasks.chat_core.humaneval import setup_humaneval_task
    from eval_tasks.chat_core.mmlu import setup_mmlu_task

    enc, _ = get_custom_tokenizer()

    chatcore_evaluator = ChatCoreEvaluator(
        model=raw_model,
        tokenizer=enc,
        device=device,
        master_process=master_process,
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        max_examples=config.chat_core_max_examples,
        num_samples=config.chat_core_num_samples,
        max_tokens=config.chat_core_max_tokens,
        temperature=config.chat_core_temperature,
        top_k=config.chat_core_top_k,
        use_kv_cache=config.use_kv_cache,
    )

    # Register evaluation tasks
    setup_gsm8k_task(
        chatcore_evaluator,
        enc,
        split="test",
        cache_dir=config.chat_core_hf_cache_dir,
    )
    setup_humaneval_task(
        chatcore_evaluator,
        enc,
        cache_dir=config.chat_core_hf_cache_dir,
        shuffle_seed=42,
    )
    setup_arc_task(
        chatcore_evaluator,
        enc,
        subset="ARC-Easy",
        split="test",
        cache_dir=config.chat_core_hf_cache_dir,
    )
    setup_arc_challenge_task(
        chatcore_evaluator,
        enc,
        subset="ARC-Challenge",
        split="test",
        cache_dir=config.chat_core_hf_cache_dir,
    )
    setup_mmlu_task(
        chatcore_evaluator,
        enc,
        subset="all",
        split="test",
        cache_dir=config.chat_core_hf_cache_dir,
    )

    return chatcore_evaluator


def setup_evaluators(
    config,
    ddp_world_size,
    ddp_rank,
    master_process,
    run_evals,
    run_core_evals,
    run_chatcore_evals,
    raw_model,
    device,
    ddp,
    generation_log_file,
    token_bytes_path,
    eval_dataloader=None,
):
    """
    Setup all evaluators based on configuration flags.

    Args:
        config: GPTConfig instance
        ddp_world_size: DDP world size
        ddp_rank: DDP rank
        master_process: Whether this is the master process
        run_evals: Whether to run training evaluations
        run_core_evals: Whether to run CORE evaluations
        run_chatcore_evals: Whether to run ChatCORE evaluations
        raw_model: The raw (unwrapped) model for evaluators
        device: Device to use
        ddp: Whether using DDP
        generation_log_file: Path to generation log file
        token_bytes_path: Path to token bytes file
        eval_dataloader: Eval dataloader provided by setup_dataloaders

    Returns:
        tuple: (evaluator, core_evaluator, chatcore_evaluator)
    """
    # Setup training evaluator
    if run_evals:
        evaluator = TrainingEvaluator(
            model=raw_model,
            eval_dataloader=eval_dataloader,
            device=device,
            master_process=master_process,
            ddp=ddp,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            generation_log_file=generation_log_file,
            token_bytes_path=token_bytes_path,
            val_loss_tokens=config.val_loss_eval_tokens,
            batch_size=config.batch_size,
            block_size=config.block_size,
            sample_seed=config.generation_seed,
            use_kv_cache=config.use_kv_cache,
            generation_verbose=config.generation_verbose,
            temperature=config.generation_temperature,
            top_k=config.generation_top_k,
            repetition_penalty=config.generation_repetition_penalty,
        )
    else:
        evaluator = None
        if master_process:
            print("Evaluations disabled - skipping eval dataloader initialization")

    # Setup CORE evaluator
    if run_core_evals:
        enc, _ = get_custom_tokenizer()
        core_evaluator = CoreEvaluator(
            model=raw_model,
            tokenizer=enc,
            device=device,
            master_process=master_process,
            ddp=ddp,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            eval_bundle_path="/mnt/localssd/VibeNanoChat/resources/eval_bundle",
            max_examples_per_task=config.core_eval_max_examples,
        )
    else:
        core_evaluator = None

    # Setup ChatCORE evaluator
    if run_chatcore_evals:
        chatcore_evaluator = setup_chatcore_evaluator(
            raw_model=raw_model,
            device=device,
            master_process=master_process,
            ddp=ddp,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
            config=config,
        )
    else:
        chatcore_evaluator = None

    return evaluator, core_evaluator, chatcore_evaluator
