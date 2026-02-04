"""Dataloader setup utilities for the trainer."""

from dataloaders.fineweb_edu_parquet_bos_dataloader import \
    FinewebEduParquetBOSDataloader
from eval_tasks import CoreEvaluator
from eval_tasks.training import TrainingEvaluator
from gpt_2.utils import get_custom_tokenizer


def create_evaluator(
    model,
    eval_dataloader,
    device,
    master_process,
    ddp,
    ddp_rank,
    ddp_world_size,
    generation_log_file,
    token_bytes_path,
    config,
):
    """
    Create a TrainingEvaluator with standard configuration.

    Args:
        model: The model to evaluate
        eval_dataloader: The evaluation dataloader to use
        device: Device to run evaluation on
        master_process: Whether this is the master process
        ddp: Whether using DDP
        ddp_rank: DDP rank
        ddp_world_size: DDP world size
        generation_log_file: Path to generation log file
        token_bytes_path: Path to token bytes file
        config: GPTConfig instance

    Returns:
        TrainingEvaluator instance
    """
    return TrainingEvaluator(
        model=model,
        eval_dataloader=eval_dataloader,
        device=device,
        master_process=master_process,
        ddp=ddp,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
        generation_log_file=generation_log_file,
        token_bytes_path=token_bytes_path,
        val_loss_steps=config.val_loss_eval_batches,
        sample_seed=config.generation_seed,
        use_kv_cache=config.use_kv_cache,
    )


def setup_pretraining_dataloaders(
    config,
    ddp_world_size,
    ddp_rank,
    master_process,
    run_evals,
    device="cuda",
):
    """
    Setup dataloaders for pretraining mode.

    Args:
        config: GPTConfig instance
        ddp_world_size: DDP world size
        ddp_rank: DDP rank
        master_process: Whether this is the master process
        run_evals: Whether to create evaluation dataloader
        device: Device for tensor allocation ('cuda' or 'cpu')

    Returns:
        tuple: (train_dataloader, eval_dataloader)
    """
    # Use Parquet BOS-aligned dataloader for pretraining
    DataloaderClass = FinewebEduParquetBOSDataloader
    data_dir = config.data_dir_pretrain_parquet
    extra_kwargs = {
        "buffer_size": config.bos_dataloader_buffer_size,
        "device": device,
        "tokenizer_threads": 4,
        "tokenizer_batch_size": 128,
    }
    if master_process:
        print("📚 PRETRAINING: Using Parquet BOS-aligned dataloader (nanochat-style)")
        print(f"   Data directory: {data_dir}")
        print("   Expected token waste: ~35% (cropping for document boundaries)")

    train_dataloader = DataloaderClass(
        data_dir=data_dir,
        batch_size=config.batch_size,
        block_size=config.block_size,
        ddp_world_size=ddp_world_size,
        ddp_rank=ddp_rank,
        split="train",
        master_process=master_process,
        **extra_kwargs,
    )

    if run_evals:
        eval_dataloader = DataloaderClass(
            data_dir=data_dir,
            batch_size=config.batch_size,
            block_size=config.block_size,
            ddp_world_size=ddp_world_size,
            ddp_rank=ddp_rank,
            split="val",
            master_process=master_process,
            **extra_kwargs,
        )
    else:
        eval_dataloader = None

    return train_dataloader, eval_dataloader


def setup_sft_dataloaders(
    config,
    ddp_world_size,
    ddp_rank,
    master_process,
    run_evals,
):
    """
    Setup dataloaders for SFT training mode.

    Args:
        config: GPTConfig instance
        ddp_world_size: DDP world size
        ddp_rank: DDP rank
        master_process: Whether this is the master process
        run_evals: Whether to create evaluation dataloader

    Returns:
        tuple: (train_dataloader, eval_dataloader)
    """
    if master_process:
        print("\n" + "=" * 80)
        print("🎯 SFT TRAINING MODE: Using Multiplex datasets")
        print("=" * 80 + "\n")

    # Import required dataloaders
    from dataloaders.arc_dataloader import ARCDataLoader
    from dataloaders.gsm8k_dataloader import GSM8KDataLoader
    from dataloaders.multiplex_dataloader import (create_multiplex_dataloader,
                                                  create_sft_collate_fn)
    from dataloaders.simplespelling_dataloader import SimpleSpellingDataLoader
    from dataloaders.smoltalk_dataloader import SmolTalkDataLoader
    from dataloaders.spellingbee_dataloader import SpellingBeeDataLoader

    cache_dir = config.sft_cache_dir

    if master_process:
        print("Loading SFT training datasets...")

    # Load training datasets
    arc_easy_data = ARCDataLoader(
        subset="ARC-Easy", split="train", cache_dir=cache_dir
    ).load_data(format_as_conversation=True)

    arc_challenge_data = ARCDataLoader(
        subset="ARC-Challenge", split="train", cache_dir=cache_dir
    ).load_data(format_as_conversation=True)

    gsm8k_data = GSM8KDataLoader(split="train", cache_dir=cache_dir).load_data(
        format_as_conversation=True
    )

    smoltalk_data = SmolTalkDataLoader(split="train", cache_dir=cache_dir).load_data(
        max_examples=10000
    )

    spelling_bee_data = SpellingBeeDataLoader(
        size=300, split="train", cache_dir=cache_dir
    ).load_data()

    simple_spelling_data = SimpleSpellingDataLoader(
        size=300, split="train", cache_dir=cache_dir
    ).load_data()

    if master_process:
        print(f"✓ Loaded {len(arc_easy_data)} ARC-Easy examples")
        print(f"✓ Loaded {len(arc_challenge_data)} ARC-Challenge examples")
        print(f"✓ Loaded {len(gsm8k_data)} GSM8K examples")
        print(f"✓ Loaded {len(smoltalk_data)} SmolTalk examples")
        print(f"✓ Loaded {len(spelling_bee_data)} SpellingBee examples")
        print(f"✓ Loaded {len(simple_spelling_data)} SimpleSpelling examples\n")

    # Setup tokenizer and collate function
    enc, _ = get_custom_tokenizer()
    pad_token_id = enc.encode("<|assistant_end|>", allowed_special="all")[0]
    collate_fn = create_sft_collate_fn(
        enc, pad_token_id, return_metadata=False, max_length=config.block_size
    )

    # Create multiplex dataloader for training
    train_dataloader = create_multiplex_dataloader(
        datasets=[
            ("arc_easy", arc_easy_data),
            ("arc_challenge", arc_challenge_data),
            ("gsm8k", gsm8k_data),
            ("smoltalk", smoltalk_data),
            ("spelling_bee", spelling_bee_data),
            ("simple_spelling", simple_spelling_data),
        ],
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        prefetch_factor=2,
        persistent_workers=True,
        sampling_strategy="proportional",
        collate_fn=collate_fn,
        ddp_rank=ddp_rank,
        ddp_world_size=ddp_world_size,
    )

    # Setup evaluation dataloader if needed
    if run_evals:
        if master_process:
            print("Loading SFT validation datasets...")

        arc_easy_val_data = ARCDataLoader(
            subset="ARC-Easy", split="validation", cache_dir=cache_dir
        ).load_data(format_as_conversation=True)

        arc_challenge_val_data = ARCDataLoader(
            subset="ARC-Challenge", split="validation", cache_dir=cache_dir
        ).load_data(format_as_conversation=True)

        gsm8k_val_data = GSM8KDataLoader(split="test", cache_dir=cache_dir).load_data(
            format_as_conversation=True, max_examples=1000
        )

        smoltalk_val_data = SmolTalkDataLoader(
            split="test", cache_dir=cache_dir
        ).load_data(max_examples=2000)

        if master_process:
            print(f"✓ Loaded {len(arc_easy_val_data)} ARC-Easy validation examples")
            print(
                f"✓ Loaded {len(arc_challenge_val_data)} ARC-Challenge validation examples"
            )
            print(f"✓ Loaded {len(gsm8k_val_data)} GSM8K validation examples")
            print(f"✓ Loaded {len(smoltalk_val_data)} SmolTalk validation examples\n")

        eval_dataloader = create_multiplex_dataloader(
            datasets=[
                ("arc_easy_val", arc_easy_val_data),
                ("arc_challenge_val", arc_challenge_val_data),
                ("gsm8k_val", gsm8k_val_data),
                ("smoltalk_val", smoltalk_val_data),
            ],
            batch_size=config.batch_size,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
            prefetch_factor=2,
            persistent_workers=True,
            sampling_strategy="proportional",
            collate_fn=collate_fn,
            ddp_rank=ddp_rank,
            ddp_world_size=ddp_world_size,
        )
    else:
        eval_dataloader = None

    return train_dataloader, eval_dataloader


def setup_dataloaders(
    sft_training,
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
    model,
    eval_dataloader,
    generation_log_file,
    token_bytes_path,
):
    """
    Initialize train and eval dataloaders based on training mode.

    Args:
        sft_training: Whether doing SFT training
        config: GPTConfig instance
        ddp_world_size: DDP world size
        ddp_rank: DDP rank
        master_process: Whether this is the master process
        run_evals: Whether to run evaluations
        run_core_evals: Whether to run CORE evaluations
        run_chatcore_evals: Whether to run ChatCORE evaluations
        raw_model: The raw (unwrapped) model
        device: Device to use
        ddp: Whether using DDP
        model: The model (possibly wrapped)
        eval_dataloader: Evaluation dataloader (will be set by this function)
        generation_log_file: Path to generation log file
        token_bytes_path: Path to token bytes file

    Returns:
        tuple: (train_dataloader, eval_dataloader, evaluator, core_evaluator, chatcore_evaluator)
    """
    # Setup train and eval dataloaders based on mode
    if sft_training:
        train_dataloader, eval_dataloader = setup_sft_dataloaders(
            config, ddp_world_size, ddp_rank, master_process, run_evals
        )
    else:
        train_dataloader, eval_dataloader = setup_pretraining_dataloaders(
            config, ddp_world_size, ddp_rank, master_process, run_evals, device
        )

    # Create evaluator if needed
    if run_evals:
        evaluator = create_evaluator(
            model,
            eval_dataloader,
            device,
            master_process,
            ddp,
            ddp_rank,
            ddp_world_size,
            generation_log_file,
            token_bytes_path,
            config,
        )
    else:
        evaluator = None
        if master_process:
            print("Evaluations disabled - skipping eval dataloader initialization")

    # Initialize CORE evaluator if requested
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

    # Initialize ChatCORE evaluator if requested
    if run_chatcore_evals:
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
    else:
        chatcore_evaluator = None

    return (
        train_dataloader,
        eval_dataloader,
        evaluator,
        core_evaluator,
        chatcore_evaluator,
    )
