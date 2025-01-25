import json
import logging
import math
import time

import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader, DistributedSampler

from mu.distributed import get_distributed_context

logger = logging.getLogger(__name__)


def set_seeds(seed=1337):
    torch.manual_seed(seed)


def get_dataloader(dataset, batch_size, shuffle=False, drop_last=True):
    distributed_context = get_distributed_context()

    logger.info(f"DataLoader({shuffle=}, {drop_last=})") if distributed_context.is_main_process else None

    if distributed_context.is_distributed:
        logger.info("Using DistributedSampler")
        sampler = DistributedSampler(dataset, shuffle=shuffle, drop_last=drop_last)
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, sampler=sampler)
    else:
        logger.info("Not using DistributedSampler")
        dataloader = DataLoader(dataset, batch_size=batch_size, drop_last=drop_last, shuffle=shuffle)
    logger.info(f"{len(dataloader)=} for {dataset=}")
    return dataloader


def get_optimizer(model, weight_decay, learning_rate, betas=None):
    if betas is None:
        betas = (0.9, 0.98)

    param_dict = {pn: p for pn, p in model.named_parameters() if p.requires_grad}

    decay_params = []
    nodecay_params = []

    for n, p in param_dict.items():
        if p.dim() >= 2:
            decay_params.append(p)
        else:
            nodecay_params.append(p)

    optim_groups = [
        {"params": decay_params, "weight_decay": weight_decay},
        {"params": nodecay_params, "weight_decay": 0.0},
    ]
    logger.info(f"Using AdamW({betas=}, lr={learning_rate}, {weight_decay=})")
    optimizer = torch.optim.AdamW(optim_groups, lr=learning_rate, betas=betas, fused=True)
    return optimizer


def _get_lr_lambda(current_step, warmup_steps, total_steps):
    if current_step <= warmup_steps:
        lr = current_step / max(1, warmup_steps)
    else:
        progress = (current_step - warmup_steps) / (total_steps - warmup_steps)
        lr = 0.5 * (1 + math.cos(math.pi * progress))
    return lr


def get_scheduler(optimizer, max_steps, warmup_steps):
    assert warmup_steps < max_steps, f"Must have {warmup_steps=} < {max_steps=}"
    logger.info(f"For scheduler, using {max_steps=} {warmup_steps=}")
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer, lr_lambda=lambda step: _get_lr_lambda(step, warmup_steps, max_steps)
    )
    return scheduler


def get_gradient_accumulation_steps(batch_size: int, global_batch_size: int):
    distributed_context = get_distributed_context()
    world_size = distributed_context.world_size

    gradient_accumulation_steps = global_batch_size / (batch_size * world_size)
    if not gradient_accumulation_steps.is_integer():
        logger.warning("global_batch_size is not divisible by batch_size * world_size")
        global_batch_size = round(gradient_accumulation_steps) * batch_size * world_size
        logger.warning(f"Rounded global_batch_size to {global_batch_size=}")
        return get_gradient_accumulation_steps(batch_size, global_batch_size)

    gradient_accumulation_steps = max(1, int(gradient_accumulation_steps))
    logger.info(f"Using {gradient_accumulation_steps=} {global_batch_size=} {batch_size=}")

    return gradient_accumulation_steps


class Trainer:
    def __init__(
        self,
        model,
        data_steps: int,
        batch_size: int = 1,
        global_batch_size: int = 1,
        max_grad_norm: float = 1.0,
        weight_decay: float = 0.01,
        learning_rate: float = 1e-4,
        warmup_steps: int = 8,
    ):
        self.model = model
        self.max_grad_norm = max_grad_norm

        self.trainer_step = 0
        self.distributed_context = get_distributed_context()

        self.gradient_accumulation_steps = get_gradient_accumulation_steps(
            batch_size=batch_size, global_batch_size=global_batch_size
        )
        train_steps = data_steps / self.gradient_accumulation_steps
        logger.info(f"Will train for {train_steps=}")

        self.optimizer = get_optimizer(self.model, weight_decay, learning_rate)
        self.scheduler = get_scheduler(self.optimizer, train_steps, warmup_steps)

        if self.distributed_context.is_distributed:
            self.ddp_model = DistributedDataParallel(self.model, device_ids=[self.distributed_context.local_rank])
        else:
            self.ddp_model = self.model

    def step_start_of_batch(self):
        torch.cuda.synchronize()
        self.start_time = time.perf_counter()
        self.global_batch_loss = 0.0
        self.tokens_processed = 0
        self.optimizer.zero_grad(set_to_none=True)

    def step_end_of_batch(self):
        norm = torch.nn.utils.clip_grad_norm_(self.ddp_model.parameters(), self.max_grad_norm)
        self.optimizer.step()
        self.scheduler.step()

        torch.cuda.synchronize()
        end_time = time.perf_counter()
        dt = end_time - self.start_time

        if self.distributed_context.is_distributed:
            dist.all_reduce(self.global_batch_loss, op=dist.ReduceOp.AVG)
            dist.all_reduce(self.tokens_processed, op=dist.ReduceOp.SUM)

        end_of_batch_metrics = {
            "tokens_per_s": self.tokens_processed.item() / dt,
            "global_batch_loss": self.global_batch_loss.item(),
            "grad_norm": norm.item(),
            "time_in_s": dt,
            "lr": self.optimizer.param_groups[0]["lr"],
        }

        self.start_time = None
        self.global_batch_loss = None
        self.tokens_processed = None

        return end_of_batch_metrics

    def step(self, batch):
        if self.trainer_step % self.gradient_accumulation_steps == 0:
            self.step_start_of_batch()

        self.ddp_model.train()
        batch = {k: v.to(self.distributed_context.device) for k, v in batch.items()}

        if self.distributed_context.is_distributed:
            self.ddp_model.require_backward_grad_sync = (self.trainer_step + 1) % self.gradient_accumulation_steps == 0

        outputs = self.ddp_model(**batch)
        loss = outputs.loss

        if torch.isnan(loss) or torch.isinf(loss):
            logger.info("Loss is NaN or Inf")
            raise ValueError("Loss is NaN or Inf")

        loss = loss / self.gradient_accumulation_steps
        self.global_batch_loss += loss.detach()
        loss.backward()

        input_ids_key = "encoder_input_ids"
        self.tokens_processed += torch.tensor(
            batch[input_ids_key].size(0) * batch[input_ids_key].size(1), device=self.distributed_context.device
        )

        end_of_batch_metrics = None
        if (self.trainer_step + 1) % self.gradient_accumulation_steps == 0:
            end_of_batch_metrics = self.step_end_of_batch()

        self.trainer_step += 1
        return outputs, end_of_batch_metrics

    def log(self, metrics, context):
        log_metrics = dict(
            tokens_per_s=round(metrics["tokens_per_s"], 2),
            loss=round(metrics["global_batch_loss"], 4),
            lr=round(metrics["lr"], 6),
            grad_norm=round(metrics["grad_norm"], 4),
            time_in_s=round(metrics["time_in_s"], 2),
        )
        log_entry = dict(
            message="training_step",
            epoch=context["epoch"],
            epoch_step=context["epoch_step"],
            global_step=context["global_step"],
            **log_metrics,
        )

        logger.info(json.dumps(log_entry))
