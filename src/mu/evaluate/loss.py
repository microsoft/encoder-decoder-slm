import json
import logging
from types import SimpleNamespace

import torch
import torch.distributed as dist
from tqdm import tqdm

from mu.distributed import get_distributed_context

logger = logging.getLogger(__name__)


class LossEvaluator:
    def __init__(self, dataname, dataloader):
        self.dataname = dataname
        self.dataloader = dataloader

    def evaluate(self, model):
        distributed_context = get_distributed_context()
        device = distributed_context.device

        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        count = 0

        with torch.no_grad():
            for i, batch in enumerate(tqdm(self.dataloader, desc="Loss Evaluation")):
                batch = {k: v.to(device) for k, v in batch.items()}
                outputs = model(**batch)
                loss = outputs.loss
                total_loss += loss
                count += 1
        average_loss = total_loss / count

        if distributed_context.is_distributed:
            torch.distributed.all_reduce(average_loss, op=dist.ReduceOp.AVG)

        output = SimpleNamespace(loss=average_loss.item())
        return output

    def log(self, output, context=None):
        if context is None:
            context = {}

        log_metrics = {
            "loss": round(output.loss, 8),
        }

        log_entry = {
            "message": "loss_evaluator_metrics",
            "dataset": self.dataname,
            **log_metrics,
            **context,
        }
        logger.info(json.dumps(log_entry))

    def evaluate_and_log(self, model, context=None):
        output = self.evaluate(model)
        if get_distributed_context().is_main_process:
            self.log(output, context)
        return output
