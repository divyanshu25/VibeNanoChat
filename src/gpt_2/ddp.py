# run trainer.py with ddp

import sys
import os

current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.append(parent_dir)

import torch.distributed as dist
import torch.multiprocessing as mp
from gpt_2.trainer import Trainer
from torch.distributed import init_process_group, destroy_process_group
import torch


def run_trainer():
    ddp = int(os.environ.get("RANK", -1)) != -1
    if ddp:
        print(f"Initializing DDP at rank: {os.environ['RANK']}")
        assert (
            torch.cuda.is_available()
        ), "CUDA is not available"  # check if cuda is available
        init_process_group(backend="nccl")  # initialize process group
        ddp_rank = int(os.environ["RANK"])  # get rank, each GPU has a different rank
        ddp_local_rank = int(os.environ["LOCAL_RANK"])  # get local rank
        ddp_world_size = int(os.environ.get("WORLD_SIZE", 1))  # get world size
        device = f"cuda:{ddp_local_rank}"  # get device
        torch.cuda.set_device(device)  # set device
        master_process = ddp_rank == 0  # check if master process
    else:
        ddp_rank = 0  # set rank
        ddp_local_rank = 0  # set local rank
        ddp_world_size = 1  # set world size
        master_process = True  # set master process
        device = "cpu"  # set device
        if torch.cuda.is_available():
            device = "cuda"  # set device to cuda if available
        if torch.backends.mps.is_available():
            device = "mps"  # set device to mps if available
        print(f"Using device: {device}")  # print device

    torch.manual_seed(42)
    if device == "cuda":
        torch.cuda.manual_seed(42)

    # print(f"I am GPU: {ddp_local_rank+1} of {ddp_world_size} with rank: {ddp_rank} and master process: {master_process}")

    try:
        trainer = Trainer(
            ddp,
            ddp_rank,
            ddp_local_rank,
            ddp_world_size,
            master_process,
            device,
            run_evals=True,
        )
        trainer.train()
    finally:
        if ddp:
            destroy_process_group()


if __name__ == "__main__":
    run_trainer()
