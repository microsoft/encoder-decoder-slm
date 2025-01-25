import logging
import os
from contextlib import contextmanager
from dataclasses import dataclass
from functools import lru_cache

import torch
import torch.distributed as dist

logger = logging.getLogger(__name__)


@dataclass
class DistributedContext:
    is_distributed: bool
    rank: int
    local_rank: int
    world_size: int
    device: str
    is_main_process: bool
    is_local_main_process: bool


@lru_cache
def get_distributed_context():
    is_distributed = int(os.environ.get("RANK", -1)) != -1
    if is_distributed:
        assert torch.cuda.is_available(), "CUDA is required for DDP"
        rank = int(os.environ["RANK"])
        local_rank = int(os.environ["LOCAL_RANK"])
        world_size = int(os.environ["WORLD_SIZE"])
        device = f"cuda:{local_rank}"
        torch.cuda.set_device(device)
        is_main_process = rank == 0
        is_local_main_process = local_rank == 0
    else:
        rank = 0
        local_rank = 0
        world_size = 1
        is_main_process = True
        is_local_main_process = True
        if torch.cuda.is_available():
            device = "cuda"
        else:
            device = "cpu"
    logger.info(f"using device: {device}")

    context = DistributedContext(
        is_distributed=is_distributed,
        rank=rank,
        local_rank=local_rank,
        world_size=world_size,
        is_main_process=is_main_process,
        is_local_main_process=is_local_main_process,
        device=device,
    )

    return context


def init_distributed():
    distributed_context = get_distributed_context()
    assert distributed_context.is_distributed, "Cannot initialize process_group without distributed context"
    device_id = torch.device(f"cuda:{distributed_context.local_rank}")
    dist.init_process_group(backend="nccl", device_id=device_id)


def destroy_distributed():
    distributed_context = get_distributed_context()
    assert distributed_context.is_distributed, "Cannot destroy process_group without distributed context"
    dist.destroy_process_group()


@contextmanager
def distributed_manager():
    init_distributed()
    yield
    destroy_distributed()
