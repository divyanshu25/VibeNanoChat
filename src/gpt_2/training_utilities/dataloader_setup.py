"""Dataloader setup utilities for the trainer."""

from dataloaders.fineweb_edu_parquet_bos_dataloader import \
    FinewebEduParquetBOSDataloader
from gpt_2.utils import get_custom_tokenizer


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
    device,
):
    """
    Initialize train and eval dataloaders based on training mode.

    Args:
        sft_training: Whether doing SFT training
        config: GPTConfig instance
        ddp_world_size: DDP world size
        ddp_rank: DDP rank
        master_process: Whether this is the master process
        run_evals: Whether to create evaluation dataloader
        device: Device to use

    Returns:
        tuple: (train_dataloader, eval_dataloader)
            - train_dataloader: Training dataloader
            - eval_dataloader: Evaluation dataloader (None if run_evals is False)
    """
    if sft_training:
        train_dataloader, eval_dataloader = setup_sft_dataloaders(
            config, ddp_world_size, ddp_rank, master_process, run_evals
        )
    else:
        # Setup pretraining dataloaders
        if master_process:
            print("📚 PRETRAINING: Using PyTorch-native BOS-aligned dataloader")
            print(f"   Data directory: {config.data_dir_pretrain_parquet}")

        # Create training dataloader
        train_dataloader = FinewebEduParquetBOSDataloader(
            data_dir=config.data_dir_pretrain_parquet,
            batch_size=config.batch_size,
            block_size=config.block_size,
            ddp_world_size=ddp_world_size,
            ddp_rank=ddp_rank,
            split="train",
            master_process=master_process,
            buffer_size=config.bos_dataloader_buffer_size,
            device=device,
            persistent_workers=config.dataloader_persistent_workers,
        )

        # Create validation dataloader if needed
        if run_evals:
            if master_process:
                print("🔄 Creating validation dataloader")
            eval_dataloader = FinewebEduParquetBOSDataloader(
                data_dir=config.data_dir_pretrain_parquet,
                batch_size=config.batch_size,
                block_size=config.block_size,
                ddp_world_size=ddp_world_size,
                ddp_rank=ddp_rank,
                split="val",
                master_process=False,
                buffer_size=config.bos_dataloader_buffer_size,
                device=device,
                persistent_workers=config.dataloader_persistent_workers,
            )
        else:
            eval_dataloader = None

    return train_dataloader, eval_dataloader
