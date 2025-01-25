import logging
import os

import torch
import torch.multiprocessing as mp

from mu.distributed import distributed_manager
from mu.log import init_logging

logger = logging.getLogger(__name__)


def torchrun_worker(rank, world_size, output_path, func, kwargs):
    os.environ["WORLD_SIZE"] = str(world_size)
    os.environ["RANK"] = str(rank)
    os.environ["LOCAL_RANK"] = str(rank)
    os.environ["MASTER_ADDR"] = "127.0.0.1"
    os.environ["MASTER_PORT"] = "29500"
    os.environ["OMP_NUM_THREADS"] = "32"

    init_logging(log_dir=output_path / f"rank={rank}")

    with distributed_manager():
        func(**kwargs)


def torchrun(output_path, function, kwargs):
    world_size = torch.cuda.device_count()
    mp.spawn(torchrun_worker, args=(world_size, output_path, function, kwargs), nprocs=world_size, join=True)
