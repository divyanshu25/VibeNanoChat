# Add gpt_2 to python path
import os
import sys

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import random
import time
from datetime import datetime

import torch

import wandb
# from dataloaders.open_webtext_dataloader import OpenWebtextDataloader
from dataloaders.fineweb_edu_dataloader import FinewebEduDataloader
from dataloaders.task_mixture_dataloader import TaskMixtureDataloader
from eval_tasks import CoreEvaluator
from eval_tasks.training import TrainingEvaluator
from gpt_2.config import GPTConfig
from gpt_2.gpt2_model import GPT
from gpt_2.sample_contexts import GENERAL_SAMPLE_CONTEXTS, SFT_SAMPLE_CONTEXTS
from gpt_2.utils import (calculate_num_iterations, get_custom_tokenizer,
                         get_lr, get_peak_flops, load_checkpoint,
                         save_checkpoint)


class Trainer:
    """
    GPT-2 Trainer class that handles model training, evaluation, and optimization.
    Implements modern training techniques like learning rate scheduling and gradient clipping.
    """

    def __init__(
        self,
        ddp,
        ddp_rank,
        ddp_local_rank,
        ddp_world_size,
        master_process,
        device,
        run_evals=False,
        run_core_evals=False,
        run_chatcore_evals=False,
        mid_training=False,
        sft_training=False,
        checkpoint_path=None,
        checkpoint_dir=None,
        token_bytes_path=None,
    ):
        """
        Initialize trainer with model configuration, data loading, and training parameters.

        Args:
            ddp: Whether to use distributed data parallel
            ddp_rank: Rank of current process
            ddp_local_rank: Local rank of current process
            ddp_world_size: Total number of processes
            master_process: Whether this is the master process
            device: Device to train on
            run_evals: Whether to run evaluations
            run_core_evals: Whether to run CORE benchmark evaluations
            run_chatcore_evals: Whether to run ChatCORE generative evaluations (GSM8K, etc.) after training
            mid_training: Whether to do mid-training (uses TaskMixture instead of pretraining data)
            sft_training: Whether to do SFT training (uses Multiplex dataloader with conversation data)
            checkpoint_path: Path to checkpoint to load (for mid-training or resuming)
            checkpoint_dir: Directory to save checkpoints (pretraining or midtraining specific)
            token_bytes_path: Path to pre-computed token_bytes.pt for BPB calculation
        """
        # Store basic config
        self.ddp = ddp
        self.ddp_rank = ddp_rank
        self.ddp_local_rank = ddp_local_rank
        self.ddp_world_size = ddp_world_size
        self.device = device
        self.master_process = master_process
        self.run_evals = run_evals
        self.run_core_evals = run_core_evals
        self.run_chatcore_evals = run_chatcore_evals
        self.mid_training = mid_training
        self.sft_training = sft_training
        self.checkpoint_path = checkpoint_path
        self.checkpoint_dir = checkpoint_dir
        self.token_bytes_path = token_bytes_path

        # Initialize start states
        self.start_step = 0
        self.start_epoch = 0
        self.start_global_step = 0

        # Select 4 random sample contexts for generation evaluation
        if self.sft_training or self.mid_training:
            self.sample_contexts = random.sample(
                SFT_SAMPLE_CONTEXTS, min(4, len(SFT_SAMPLE_CONTEXTS))
            )
        else:
            self.sample_contexts = random.sample(
                GENERAL_SAMPLE_CONTEXTS, min(4, len(GENERAL_SAMPLE_CONTEXTS))
            )

        # Setup components
        self._setup_logging()
        self._setup_model()
        self._setup_hyperparameters()
        self._setup_dataloaders()
        self._setup_optimizer_and_checkpoint()
        self._setup_wandb()
        # self._setup_token_bytes()

    def _setup_logging(self):
        """Setup generation log file for tracking model outputs."""
        if self.master_process:
            log_dir = "logs"
            os.makedirs(log_dir, exist_ok=True)
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.generation_log_file = os.path.join(
                log_dir, f"generations_{timestamp}.txt"
            )
            print(f"📝 Saving generations to: {self.generation_log_file}")
        else:
            self.generation_log_file = None

    def _setup_model(self):
        """Initialize GPT model and wrap with DDP if needed."""
        self.config = GPTConfig()

        # Create raw model and keep reference BEFORE any wrapping
        # This reference will always point to the unwrapped model with updated weights
        self.raw_model = GPT(self.config)
        self.raw_model.to(self.device)

        # Wrap with torch.compile for faster training
        self.model = torch.compile(self.raw_model)

        # Wrap with DDP for distributed training
        if self.ddp:
            self.model = torch.nn.parallel.DistributedDataParallel(
                self.model, device_ids=[self.ddp_local_rank]
            )

        # Note: self.raw_model stays unchanged and shares parameters with self.model
        # This allows clean checkpoint saving without unwrapping

    def _setup_hyperparameters(self):
        """Configure training hyperparameters based on training mode."""
        self.num_epochs = self.config.num_epochs
        self.run_evals_after = self.config.eval_interval  # every 100 steps

        # Batch size and gradient accumulation
        if self.sft_training:
            # For SFT, we count examples (conversations) not tokens
            # No gradient accumulation - process each batch immediately
            self.grad_accumulation_steps = 1
            self.total_batch_size = self.config.batch_size * self.ddp_world_size
        else:
            # For pretrain/midtrain, we count tokens
            self.total_batch_size = self.config.total_batch_size
            self.grad_accumulation_steps = self.total_batch_size // (
                self.config.batch_size * self.config.block_size * self.ddp_world_size
            )

            assert (
                self.total_batch_size
                % (
                    self.config.batch_size
                    * self.config.block_size
                    * self.ddp_world_size
                )
                == 0
            ), "Total batch size must be divisible by batch_size * block_size * world_size"

        if self.master_process:
            print(f"Total batch size: {self.total_batch_size}")
            print(f"Grad accumulation steps: {self.grad_accumulation_steps}")

        # Learning rate scheduling parameters
        self.max_learning_rate = self.config.max_learning_rate
        self.min_learning_rate = self.max_learning_rate * self.config.min_lr_ratio

        # Automatically calculate steps based on config settings for all phases
        if self.master_process:
            print("\n" + "=" * 80)
            if self.sft_training:
                print("📊 CALCULATING SFT TRAINING STEPS")
            elif self.mid_training:
                print("📊 CALCULATING MID-TRAINING STEPS")
            else:
                print("📊 CALCULATING PRETRAINING STEPS")
            print("=" * 80)

        num_iterations, flops_per_token, _ = calculate_num_iterations(
            self.raw_model, self.config, self.master_process
        )
        self.max_steps = num_iterations
        self.flops_per_token = flops_per_token

        if self.master_process:
            print("=" * 80 + "\n")

        # Initialize peak FLOPs for MFU calculation
        if self.device.startswith("cuda"):
            device_name = torch.cuda.get_device_name(self.device)
            self.peak_flops = get_peak_flops(device_name)
            if self.master_process:
                print(f"GPU: {device_name}")
                print(f"Peak FLOPS (BF16): {self.peak_flops:.2e}\n")
        else:
            self.peak_flops = float("inf")  # MFU not meaningful for non-CUDA devices

        # Set warmup steps based on training phase (calculated as % of max_steps)
        if self.sft_training:
            self.warmup_steps = int(self.max_steps * self.config.lr_warmup_ratio_sft)
        elif self.mid_training:
            self.warmup_steps = int(
                self.max_steps * self.config.lr_warmup_ratio_midtrain
            )
        else:
            self.warmup_steps = int(
                self.max_steps * self.config.lr_warmup_ratio_pretrain
            )

    def _create_evaluator(self, eval_dataloader):
        """Create a TrainingEvaluator with standard configuration.

        Args:
            eval_dataloader: The evaluation dataloader to use

        Returns:
            TrainingEvaluator instance
        """
        return TrainingEvaluator(
            model=self.model,
            eval_dataloader=eval_dataloader,
            device=self.device,
            master_process=self.master_process,
            ddp=self.ddp,
            ddp_rank=self.ddp_rank,
            ddp_world_size=self.ddp_world_size,
            generation_log_file=self.generation_log_file,
            token_bytes_path=self.token_bytes_path,
            val_loss_steps=self.config.val_loss_eval_batches,
            sample_seed=self.config.generation_seed,
            use_kv_cache=self.config.use_kv_cache,
        )

    def _setup_dataloaders(self):
        """Initialize train and eval dataloaders based on training mode."""
        if self.sft_training:
            self._setup_sft_dataloaders()
        elif self.mid_training:
            self._setup_midtraining_dataloaders()
        else:
            self._setup_pretraining_dataloaders()

        # Unified evaluator creation for all modes
        if self.run_evals:
            self.evaluator = self._create_evaluator(self.eval_dataloader)
        else:
            self.evaluator = None
            if self.master_process:
                print("Evaluations disabled - skipping eval dataloader initialization")

        # Initialize CORE evaluator if requested
        if self.run_core_evals:
            enc, _ = get_custom_tokenizer()
            self.core_evaluator = CoreEvaluator(
                model=self.raw_model,
                tokenizer=enc,
                device=self.device,
                master_process=self.master_process,
                ddp=self.ddp,
                ddp_rank=self.ddp_rank,
                ddp_world_size=self.ddp_world_size,
                eval_bundle_path="/mnt/localssd/NanoGPT/resources/eval_bundle",
                # tasks_to_run=core_tasks_subset,
                max_examples_per_task=self.config.core_eval_max_examples,
            )
        else:
            self.core_evaluator = None

        # Initialize ChatCORE evaluator if requested (for generative evaluation)
        if self.run_chatcore_evals:
            from eval_tasks.chat_core.arc_challenge import \
                setup_arc_challenge_task
            from eval_tasks.chat_core.arc_easy import setup_arc_task
            from eval_tasks.chat_core.evaluator import ChatCoreEvaluator
            from eval_tasks.chat_core.gsm8k import setup_gsm8k_task
            from eval_tasks.chat_core.humaneval import setup_humaneval_task
            from eval_tasks.chat_core.mmlu import setup_mmlu_task

            enc, _ = get_custom_tokenizer()
            self.chatcore_evaluator = ChatCoreEvaluator(
                model=self.raw_model,
                tokenizer=enc,
                device=self.device,
                master_process=self.master_process,
                ddp=self.ddp,
                ddp_rank=self.ddp_rank,
                ddp_world_size=self.ddp_world_size,
                max_examples=self.config.chat_core_max_examples,  # Use full test set for final evaluation
                num_samples=self.config.chat_core_num_samples,
                max_tokens=self.config.chat_core_max_tokens,
                temperature=self.config.chat_core_temperature,
                top_k=self.config.chat_core_top_k,
                use_kv_cache=self.config.use_kv_cache,
            )

            # Register evaluation tasks
            setup_gsm8k_task(
                self.chatcore_evaluator,
                enc,
                split="test",
                cache_dir=self.config.chat_core_hf_cache_dir,
            )
            setup_humaneval_task(
                self.chatcore_evaluator,
                enc,
                cache_dir=self.config.chat_core_hf_cache_dir,
                shuffle_seed=42,
            )
            setup_arc_task(
                self.chatcore_evaluator,
                enc,
                subset="ARC-Easy",
                split="test",
                cache_dir=self.config.chat_core_hf_cache_dir,
            )
            setup_arc_challenge_task(
                self.chatcore_evaluator,
                enc,
                subset="ARC-Challenge",
                split="test",
                cache_dir=self.config.chat_core_hf_cache_dir,
            )
            setup_mmlu_task(
                self.chatcore_evaluator,
                enc,
                subset="all",
                split="test",
                cache_dir=self.config.chat_core_hf_cache_dir,
            )
        else:
            self.chatcore_evaluator = None

    def _setup_sft_dataloaders(self):
        """Setup dataloaders for SFT training mode."""
        if self.master_process:
            print("\n" + "=" * 80)
            print("🎯 SFT TRAINING MODE: Using Multiplex datasets")
            print("=" * 80 + "\n")

        # Import required dataloaders
        from dataloaders.arc_dataloader import ARCDataLoader
        from dataloaders.gsm8k_dataloader import GSM8KDataLoader
        from dataloaders.multiplex_dataloader import (
            create_multiplex_dataloader, create_sft_collate_fn)
        from dataloaders.simplespelling_dataloader import \
            SimpleSpellingDataLoader
        from dataloaders.smoltalk_dataloader import SmolTalkDataLoader
        from dataloaders.spellingbee_dataloader import SpellingBeeDataLoader

        cache_dir = self.config.sft_cache_dir

        if self.master_process:
            print("Loading SFT training datasets...")

        # Load datasets exactly as specified in lines 558-619 of multiplex_dataloader.py
        # 1. ARC-Easy
        arc_easy_data = ARCDataLoader(
            subset="ARC-Easy", split="train", cache_dir=cache_dir
        ).load_data(format_as_conversation=True)

        # 2. ARC-Challenge
        arc_challenge_data = ARCDataLoader(
            subset="ARC-Challenge", split="train", cache_dir=cache_dir
        ).load_data(format_as_conversation=True)

        # 3. GSM8K
        gsm8k_data = GSM8KDataLoader(split="train", cache_dir=cache_dir).load_data(
            format_as_conversation=True
        )

        # 4. SmolTalk
        smoltalk_data = SmolTalkDataLoader(
            split="train", cache_dir=cache_dir
        ).load_data(max_examples=10000)

        # 5. SpellingBee
        spelling_bee_data = SpellingBeeDataLoader(
            size=300, split="train", cache_dir=cache_dir
        ).load_data()

        # 6. SimpleSpelling
        simple_spelling_data = SimpleSpellingDataLoader(
            size=300, split="train", cache_dir=cache_dir
        ).load_data()

        if self.master_process:
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
            enc, pad_token_id, return_metadata=False, max_length=self.config.block_size
        )

        # Create multiplex dataloader for training
        self.train_dataloader = create_multiplex_dataloader(
            datasets=[
                ("arc_easy", arc_easy_data),
                ("arc_challenge", arc_challenge_data),
                ("gsm8k", gsm8k_data),
                ("smoltalk", smoltalk_data),
                ("spelling_bee", spelling_bee_data),
                ("simple_spelling", simple_spelling_data),
            ],
            batch_size=self.config.batch_size,
            shuffle=True,
            num_workers=4,  # Parallel data loading workers
            pin_memory=True,  # Faster GPU transfer
            prefetch_factor=2,  # Prefetch 2 batches per worker
            persistent_workers=True,  # Keep workers alive between epochs
            sampling_strategy="proportional",
            collate_fn=collate_fn,
            ddp_rank=self.ddp_rank,  # DDP rank for sharding
            ddp_world_size=self.ddp_world_size,  # DDP world size for sharding
        )

        # Validation: Multiple task validation sets to match training distribution
        if self.run_evals:
            if self.master_process:
                print("Loading SFT validation datasets...")

            # Load validation splits from all training tasks
            # Note: ARC has 'train'/'validation'/'test', but GSM8K and SmolTalk only have 'train'/'test'
            # We use 'validation' for ARC (proper dev set) and 'test' for others (no validation split exists)
            arc_easy_val_data = ARCDataLoader(
                subset="ARC-Easy", split="validation", cache_dir=cache_dir
            ).load_data(format_as_conversation=True)

            arc_challenge_val_data = ARCDataLoader(
                subset="ARC-Challenge", split="validation", cache_dir=cache_dir
            ).load_data(format_as_conversation=True)

            # GSM8K only has train/test splits, so use test for validation
            gsm8k_val_data = GSM8KDataLoader(
                split="test", cache_dir=cache_dir
            ).load_data(format_as_conversation=True, max_examples=1000)

            # SmolTalk only has train/test splits, so use test for validation
            smoltalk_val_data = SmolTalkDataLoader(
                split="test", cache_dir=cache_dir
            ).load_data(max_examples=2000)

            # Note: No validation splits for SpellingBee/SimpleSpelling
            # (they're generated, so train=test essentially)

            if self.master_process:
                print(f"✓ Loaded {len(arc_easy_val_data)} ARC-Easy validation examples")
                print(
                    f"✓ Loaded {len(arc_challenge_val_data)} ARC-Challenge validation examples"
                )
                print(f"✓ Loaded {len(gsm8k_val_data)} GSM8K validation examples")
                print(
                    f"✓ Loaded {len(smoltalk_val_data)} SmolTalk validation examples\n"
                )

            self.eval_dataloader = create_multiplex_dataloader(
                datasets=[
                    ("arc_easy_val", arc_easy_val_data),
                    ("arc_challenge_val", arc_challenge_val_data),
                    ("gsm8k_val", gsm8k_val_data),
                    ("smoltalk_val", smoltalk_val_data),
                ],
                batch_size=self.config.batch_size,
                shuffle=False,
                num_workers=2,  # Fewer workers for validation (less frequent)
                pin_memory=True,  # Faster GPU transfer
                prefetch_factor=2,  # Prefetch 2 batches per worker
                persistent_workers=True,  # Keep workers alive between epochs
                sampling_strategy="proportional",  # Match training distribution
                collate_fn=collate_fn,
                ddp_rank=self.ddp_rank,  # DDP rank for sharding
                ddp_world_size=self.ddp_world_size,  # DDP world size for sharding
            )
        else:
            self.eval_dataloader = None

    def _setup_midtraining_dataloaders(self):
        """Setup dataloaders for mid-training mode."""
        DataloaderClass = TaskMixtureDataloader
        data_dir = self.config.data_dir_midtrain

        if self.master_process:
            print("\n" + "=" * 80)
            print("🔄 MID-TRAINING MODE: Using TaskMixture datasets")
            print("=" * 80 + "\n")

        self.train_dataloader = DataloaderClass(
            data_dir=data_dir,
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
            ddp_world_size=self.ddp_world_size,
            ddp_rank=self.ddp_rank,
            split="train",
            master_process=self.master_process,
        )

        if self.run_evals:
            self.eval_dataloader = DataloaderClass(
                data_dir=data_dir,
                batch_size=self.config.batch_size,
                block_size=self.config.block_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                split="val",
                master_process=self.master_process,
            )
        else:
            self.eval_dataloader = None

    def _setup_pretraining_dataloaders(self):
        """Setup dataloaders for pretraining mode."""
        DataloaderClass = FinewebEduDataloader
        data_dir = self.config.data_dir_pretrain

        self.train_dataloader = DataloaderClass(
            data_dir=data_dir,
            batch_size=self.config.batch_size,
            block_size=self.config.block_size,
            ddp_world_size=self.ddp_world_size,
            ddp_rank=self.ddp_rank,
            split="train",
            master_process=self.master_process,
        )

        if self.run_evals:
            self.eval_dataloader = DataloaderClass(
                data_dir=data_dir,
                batch_size=self.config.batch_size,
                block_size=self.config.block_size,
                ddp_world_size=self.ddp_world_size,
                ddp_rank=self.ddp_rank,
                split="val",
                master_process=self.master_process,
            )
        else:
            self.eval_dataloader = None

    def _setup_optimizer_and_checkpoint(self):
        """Initialize optimizer and load checkpoint if provided."""
        self.optimizer = self.raw_model.configure_optimizers(
            learning_rate=self.max_learning_rate,
            weight_decay=self.config.weight_decay,
            device=self.device,
        )

        if self.checkpoint_path:
            # Determine checkpoint source
            is_pretrain_ckpt = "pretrain_checkpoints" in self.checkpoint_path
            is_midtrain_ckpt = "midtrain_checkpoints" in self.checkpoint_path
            is_sft_ckpt = "sft_checkpoints" in self.checkpoint_path

            # Define training scenario flags
            is_rollover_pretrain_to_midtrain = self.mid_training and is_pretrain_ckpt
            is_rollover_midtrain_to_sft = self.sft_training and is_midtrain_ckpt
            is_resume_pretrain = (
                not self.mid_training and not self.sft_training and is_pretrain_ckpt
            )
            is_resume_midtrain = self.mid_training and is_midtrain_ckpt
            is_resume_sft = self.sft_training and is_sft_ckpt

            # Load optimizer only when resuming (not when rolling over)
            is_rollover = (
                is_rollover_pretrain_to_midtrain or is_rollover_midtrain_to_sft
            )
            should_load_optimizer = not is_rollover
            # Don't print resume info for rollover scenarios
            should_print_resume_info = not is_rollover

            checkpoint_result = load_checkpoint(
                checkpoint_path=self.checkpoint_path,
                model=self.raw_model,
                device=self.device,
                optimizer=self.optimizer if should_load_optimizer else None,
                master_process=self.master_process,
                print_resume_info=should_print_resume_info,
            )
            if checkpoint_result["config"]:
                self.config = checkpoint_result["config"]

            # Reset training counters only when rolling over
            if is_rollover_pretrain_to_midtrain:
                self.start_epoch = 0
                self.start_step = 0
                self.start_global_step = 0
                if self.master_process:
                    print(
                        "🔄 Rollover: Pretraining → Mid-training "
                        "(weights loaded, fresh optimizer, counters reset to 0)"
                    )
                    print("   Training will start from global_step: 0")
                    print(f"{'='*80}\n")
            elif is_rollover_midtrain_to_sft:
                self.start_epoch = 0
                self.start_step = 0
                self.start_global_step = 0
                if self.master_process:
                    print(
                        "🔄 Rollover: Mid-training → SFT "
                        "(weights loaded, fresh optimizer, counters reset to 0)"
                    )
                    print("   Training will start from global_step: 0")
                    print(f"{'='*80}\n")
            # Keep checkpoint counters when resuming
            elif is_resume_pretrain or is_resume_midtrain or is_resume_sft:
                self.start_epoch = checkpoint_result["start_epoch"]
                self.start_step = checkpoint_result["start_step"]
                self.start_global_step = checkpoint_result["start_global_step"]

                if self.master_process:
                    if is_resume_pretrain:
                        mode = "pretraining"
                    elif is_resume_midtrain:
                        mode = "mid-training"
                    else:
                        mode = "SFT"
                    print(
                        f"🔄 Resuming {mode} from epoch {self.start_epoch}, "
                        f"step {self.start_step}, global_step {self.start_global_step} "
                        "(weights + optimizer loaded)"
                    )
            else:
                # Abort if checkpoint scenario is not recognized
                raise ValueError(
                    f"Unrecognized checkpoint scenario!\n"
                    f"  Checkpoint path: {self.checkpoint_path}\n"
                    f"  mid_training flag: {self.mid_training}\n"
                    f"  sft_training flag: {self.sft_training}\n"
                    f"  is_pretrain_ckpt: {is_pretrain_ckpt}\n"
                    f"  is_midtrain_ckpt: {is_midtrain_ckpt}\n"
                    f"  is_sft_ckpt: {is_sft_ckpt}\n\n"
                    f"Expected checkpoint path patterns:\n"
                    f"  - For rollover pretrain→midtrain: mid_training=True + 'pretrain_checkpoints' in path\n"
                    f"  - For rollover midtrain→sft: sft_training=True + 'midtrain_checkpoints' in path\n"
                    f"  - For resume pretraining: mid_training=False + sft_training=False + 'pretrain_checkpoints' in path\n"
                    f"  - For resume mid-training: mid_training=True + 'midtrain_checkpoints' in path\n"
                    f"  - For resume SFT: sft_training=True + 'sft_checkpoints' in path"
                )

            # Handle epoch boundary for resumed checkpoints
            if self.start_step >= self.max_steps:
                self.start_step = 0
                self.start_epoch += 1

    def _setup_wandb(self):
        """Initialize Weights & Biases for experiment tracking."""
        if self.master_process:
            if self.sft_training:
                project_name = "gpt2-sft"
                training_mode = "SFT"
            elif self.mid_training:
                project_name = "gpt2-midtraining"
                training_mode = "mid-training"
            else:
                project_name = "gpt2-pretraining"
                training_mode = "pretraining"

            # Calculate total FLOPs budget for this run
            total_flops = self.flops_per_token * self.total_batch_size * self.max_steps

            # Format FLOPs in scientific notation (e.g., "7.9e18" -> "7.9e18")
            flops_str = f"{total_flops:.1e}"

            # Create run name: L{layers}-{flops}
            # Example: "L12-7.9e18" for 12 layers and 7.9e18 FLOPs
            run_name = f"model_L{self.config.n_layer}-{flops_str}"

            wandb.init(
                project=project_name,
                name=run_name,
                config={
                    "model_type": "GPT-2",
                    "training_mode": training_mode,
                    "batch_size": self.config.batch_size,
                    "block_size": self.config.block_size,
                    "max_learning_rate": self.max_learning_rate,
                    "min_learning_rate": self.min_learning_rate,
                    "warmup_steps": self.warmup_steps,
                    "max_steps": self.max_steps,
                    "num_epochs": self.num_epochs,
                    "weight_decay": self.config.weight_decay,
                    "gradient_clip_norm": self.config.gradient_clip_norm,
                    "run_evals": self.run_evals,
                    "run_core_evals": self.run_core_evals,
                    "run_chatcore_evals": self.run_chatcore_evals,
                    "start_step": self.start_step,
                    "n_layers": self.config.n_layer,
                    "total_flops": total_flops,
                },
            )

    def _setup_token_bytes(self):
        """Load pre-computed token_bytes tensor for BPB calculation."""
        if self.token_bytes_path is not None:
            self._token_bytes = torch.load(self.token_bytes_path, weights_only=True).to(
                self.device
            )
        else:
            self._token_bytes = None
            if self.master_process:
                print(
                    "⚠️  token_bytes_path not provided - train BPB will not be computed"
                )

    def train(self):
        """
        Main training loop that implements the full training procedure.
        Includes gradient clipping, learning rate scheduling, and progress monitoring.
        """
        ## Start training ##
        # Set precision for matrix multiplications (improves performance on modern GPUs)
        torch.set_float32_matmul_precision("high")

        # Calculate total steps once for the entire training run
        total_steps = self.max_steps * self.num_epochs

        if self.master_process:
            # Calculate FLOPs statistics (using pre-computed values from init)
            total_tokens = self.total_batch_size * total_steps
            total_flops = self.flops_per_token * total_tokens
            num_params = self.raw_model.num_scaling_params()
            tokens_params_ratio = total_tokens / num_params

            print("\n" + "=" * 80)
            if self.mid_training:
                print("🔄 STARTING MID-TRAINING")
            else:
                print("🚀 STARTING TRAINING")
            print("=" * 80)
            if self.start_global_step > 0:
                print(
                    f"📍 Resuming from epoch {self.start_epoch}, step {self.start_step} "
                    f"(global step {self.start_global_step:,})"
                )
            print(f"📊 Steps per epoch: {self.max_steps:,}")
            print(f"📊 Total epochs: {self.num_epochs}")
            print(f"📊 Total steps: {total_steps:,}")
            print(
                f"📦 Batch size: {self.config.batch_size} x {self.config.block_size} tokens"
            )
            print(f"🌐 World size: {self.ddp_world_size} GPUs")
            print(f"🎯 Total batch size: {self.total_batch_size:,} tokens/step")
            print(f"🔢 Model parameters: {num_params:,}")
            print(f"💫 FLOPs per token: {self.flops_per_token:.3e}")
            print(f"📈 Total training tokens: {total_tokens:,}")
            print(f"⚡ Total training FLOPs: {total_flops:.3e}")
            print(f"📐 Tokens:Params ratio: {tokens_params_ratio:.2f}")
            print("=" * 80 + "\n")

        # Main training loop over epochs
        global_step = self.start_global_step  # Track global step across all epochs
        for epoch in range(self.start_epoch, self.num_epochs):
            # Process all batches in the current epoch
            # Only use start_step for the first resumed epoch, then start from 0
            epoch_start_step = self.start_step if epoch == self.start_epoch else 0

            # Unified training loop for all modes (pretrain/midtrain/sft)
            for step in range(epoch_start_step, self.max_steps):
                start_time = time.time()  # Track step timing
                self.optimizer.zero_grad()
                loss_accumulator = torch.tensor(0.0, device=self.device)
                # # BPB accumulators
                # torch.tensor(0.0, dtype=torch.float32, device=self.device)
                # torch.tensor(0, dtype=torch.int64, device=self.device)

                # Track active tokens for SFT training (where targets >= 0)
                num_active_tokens = torch.tensor(
                    0, dtype=torch.int64, device=self.device
                )

                for micro_step in range(self.grad_accumulation_steps):
                    # Get training batch and move to device
                    x, y = self.train_dataloader.next_batch()
                    x, y = x.to(self.device), y.to(self.device)

                    # Forward pass: compute per-token loss
                    with torch.autocast(device_type=self.device, dtype=torch.bfloat16):
                        logits, per_token_loss = self.model(x, y, loss_reduction="none")

                    # Compute mean loss for backprop
                    loss = per_token_loss.mean() / self.grad_accumulation_steps
                    loss_accumulator += loss

                    # Count active tokens (for SFT: targets >= 0, for pretrain: all tokens)
                    if self.sft_training:
                        num_active_tokens += (y >= 0).sum()

                    # Accumulate for BPB calculation
                    # if self._token_bytes is not None:
                    #     nats, bytes = accumulate_bpb(per_token_loss, y, self._token_bytes)
                    #     total_nats += nats
                    #     total_bytes += bytes

                    if self.ddp:
                        self.model.require_backward_grad_sync = (
                            micro_step == self.grad_accumulation_steps - 1
                        )
                    # Backward pass: compute gradients
                    loss.backward()

                if self.ddp:
                    torch.distributed.all_reduce(
                        loss_accumulator, op=torch.distributed.ReduceOp.AVG
                    )
                    # Sum active tokens across all ranks for SFT
                    if self.sft_training:
                        torch.distributed.all_reduce(
                            num_active_tokens, op=torch.distributed.ReduceOp.SUM
                        )
                    # if self._token_bytes is not None:
                    #     torch.distributed.all_reduce(
                    #         total_nats, op=torch.distributed.ReduceOp.SUM
                    #     )
                    #     torch.distributed.all_reduce(
                    #         total_bytes, op=torch.distributed.ReduceOp.SUM
                    #     )

                # Gradient clipping to prevent exploding gradients
                # This stabilizes training by limiting gradient magnitude
                norm = torch.nn.utils.clip_grad_norm_(
                    self.model.parameters(), self.config.gradient_clip_norm
                )

                # Update learning rate based on global step (continuous across epochs)
                lr = get_lr(
                    global_step=global_step,
                    warmup_steps=self.warmup_steps,
                    max_steps=self.max_steps,
                    num_epochs=self.num_epochs,
                    max_learning_rate=self.max_learning_rate,
                    min_learning_rate=self.min_learning_rate,
                )
                for param_group in self.optimizer.param_groups:
                    param_group["lr"] = lr

                # Apply gradients to update model parameters
                self.optimizer.step()

                # Synchronize CUDA operations for accurate timing
                if self.device == "cuda":
                    torch.cuda.synchronize()
                end_time = time.time()

                # Calculate training throughput (tokens processed per second)
                if self.sft_training:
                    # For SFT: count only active tokens (where targets >= 0)
                    tokens_per_second = num_active_tokens.item() / (
                        end_time - start_time
                    )
                else:
                    # For pretrain/midtrain: all tokens in batch are active
                    tokens_per_second = (
                        self.config.batch_size
                        * self.config.block_size
                        * self.grad_accumulation_steps
                        * self.ddp_world_size
                        / (end_time - start_time)
                    )

                # Calculate FLOPs metrics
                flops_per_second = self.flops_per_token * tokens_per_second
                flops_so_far = (
                    self.flops_per_token * self.total_batch_size * global_step
                )
                # MFU: Model FLOPs Utilization (% of theoretical peak performance)
                mfu = 100 * flops_per_second / (self.peak_flops * self.ddp_world_size)

                # Periodically estimate loss on train/val sets for monitoring
                val_loss = None  # Will be set if evals run
                should_eval = (
                    global_step % self.run_evals_after == 0
                    or global_step == total_steps - 1
                )
                if self.run_evals and should_eval:
                    val_loss = self.evaluator.estimate_validation_loss(
                        step=step, global_step=global_step, total_flops=flops_so_far
                    )

                    # self.evaluator.sample_from_model(
                    #     num_sequences=self.config.generation_num_samples,
                    #     max_length=1024,
                    #     context=sample_context,
                    #     step=step,
                    # )

                # Run CORE evaluations if enabled
                if self.run_core_evals and should_eval:
                    self.core_evaluator.evaluate_all_tasks(
                        step=step, global_step=global_step
                    )

                # Save checkpoint at intervals or at end of training (independent of evals)
                if self.sft_training:
                    checkpoint_interval = self.config.checkpoint_interval_sft
                elif self.mid_training:
                    checkpoint_interval = self.config.checkpoint_interval_midtrain
                else:
                    checkpoint_interval = self.config.checkpoint_interval_pretrain

                save_checkpoint(
                    model=self.raw_model,  # Pass unwrapped model for clean state_dict
                    optimizer=self.optimizer,
                    step=step,
                    epoch=epoch,
                    global_step=global_step,
                    val_loss=val_loss,
                    checkpoint_dir=self.checkpoint_dir,
                    ddp=self.ddp,
                    checkpoint_interval=checkpoint_interval,
                    max_steps=self.max_steps,
                    num_epochs=self.num_epochs,
                    master_process=self.master_process,
                    mid_training=self.mid_training,
                    sft_training=self.sft_training,
                )

                # Log metrics to wandb
                if self.master_process:
                    train_loss = loss_accumulator.item()
                    # # Compute train BPB
                    # total_bytes_val = total_bytes.item()
                    # if total_bytes_val > 0:
                    #     train_bpb = total_nats.item() / (math.log(2) * total_bytes_val)
                    # else:
                    #     train_bpb = float("inf")

                    wandb.log(
                        {
                            # "epoch": epoch,
                            # "epoch_step": step,
                            "step": global_step,
                            "train_loss": train_loss,
                            # "train_bpb": train_bpb,
                            "learning_rate": lr,
                            "tokens_per_second": tokens_per_second,
                            "time_taken": end_time - start_time,
                            "gradient_norm": norm,
                            "flops_per_second": flops_per_second,
                            # "total_training_flops": flops_so_far,
                            "mfu": mfu,
                        }
                    )

                    # Print comprehensive training statistics
                    progress = (global_step + 1) / total_steps * 100
                    print(
                        f"[Epoch {epoch+1}/{self.num_epochs}] [Step {step:>5}/{self.max_steps}] ({progress:>5.1f}%) | "
                        f"Loss: {train_loss:.4f} | "
                        f"LR: {lr:.2e} | "
                        f"Grad: {norm:.2e} | "
                        f"Speed: {tokens_per_second/1000:.1f}K tok/s | "
                        f"MFU: {mfu:.2f}% | "
                        f"Time: {end_time - start_time:.2f}s"
                    )

                # Increment global step counter
                global_step += 1

            # Run ChatCORE evaluation after each epoch
            if self.run_chatcore_evals:
                if self.master_process:
                    print("\n" + "=" * 80)
                    print(
                        f"🎯 Running ChatCORE evaluation after Epoch {epoch+1}/{self.num_epochs}..."
                    )
                    print("=" * 80 + "\n")

                chatcore_results = self.chatcore_evaluator.evaluate_all_tasks(
                    step=step, global_step=global_step
                )

                if self.master_process:
                    print("\n" + "=" * 80)
                    print(f"📊 ChatCORE RESULTS (Epoch {epoch+1}/{self.num_epochs}):")
                    for task_name, results in chatcore_results.items():
                        accuracy = results["accuracy"]
                        correct = results["correct"]
                        total = results["total"]
                        print(f"   {task_name}: {accuracy:.2%} ({correct}/{total})")
                    print("=" * 80 + "\n")

        # Training complete
        if self.master_process:
            print("\n" + "=" * 80)
            print("✅ TRAINING COMPLETE!")
            print("=" * 80 + "\n")

        # Finish wandb run
        wandb.finish()
