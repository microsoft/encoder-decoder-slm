import json
import logging
from types import SimpleNamespace
from typing import Optional

import torch
import torch.distributed as dist
from tqdm import tqdm

from mu.distributed import get_distributed_context

logger = logging.getLogger(__name__)


class GenerateEvaluator:
    def __init__(
        self,
        dataloader,
        tokenizer,
        max_new_tokens,
        temperature=1.0,
        top_k: Optional[int] = None,
    ):
        log_args = {k: v for k, v in locals().items() if k not in ["self", "dataloader", "tokenizer"]}
        logger.info(f"GenerateEvaluator({json.dumps(log_args)})")
        self.tokenizer = tokenizer
        self.max_new_tokens = max_new_tokens
        self.dataloader = dataloader
        self.temperature = temperature
        self.top_k = top_k

    def evaluate(self, model):
        distributed_context = get_distributed_context()
        world_size = distributed_context.world_size
        rank = distributed_context.rank

        model.eval()

        input_texts = []
        generated_texts = []

        progress_bar = tqdm(self.dataloader, desc="Generate Evaluation")
        with torch.no_grad():
            for batch in progress_bar:
                input_ids = batch["encoder_input_ids"]
                input_ids = input_ids.to(distributed_context.device)
                generated_ids = model.generate(
                    input_ids=input_ids,
                    max_new_tokens=self.max_new_tokens,
                    decoder_start_token_id=self.tokenizer.bos_token_id,
                    temperature=self.temperature,
                    top_k=self.top_k,
                    eos_token_id=self.tokenizer.eos_token_id,
                    bos_token_id=self.tokenizer.bos_token_id,
                )
                generated_text = self.tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
                generated_texts.extend(generated_text)

                input_text = self.tokenizer.batch_decode(input_ids, skip_special_tokens=True)
                input_texts.extend(input_text)

        if distributed_context.is_distributed:
            generated_texts_gathered = [None for _ in range(world_size)]
            input_texts_gathered = [None for _ in range(world_size)]
            dist.barrier()

            dist.all_gather_object(generated_texts_gathered, generated_texts)
            dist.all_gather_object(input_texts_gathered, input_texts)
        else:
            generated_texts_gathered = [generated_texts]
            input_texts_gathered = [input_texts]

        if rank == 0:
            flattened_generated_texts = [g for gg in generated_texts_gathered for g in gg]
            flattened_inputs = [i for ii in input_texts_gathered for i in ii]

            output = SimpleNamespace(generations=flattened_generated_texts, inputs=flattened_inputs)
            return output
        else:
            output = SimpleNamespace(generations=None, inputs=None)
            return output

    def log(self, output, context=None):
        if context is None:
            context = {}

        generated_texts = output.generations
        input_texts = output.inputs

        if generated_texts is None or input_texts is None:
            return

        assert len(generated_texts) == len(input_texts), f"{len(generated_texts)=} != {len(input_texts)=}"
        for input_text, generated_text in zip(input_texts, generated_texts):
            to_log = dict(
                message="generate_evaluation",
                input=input_text,
                generated=generated_text,
                **context,
            )
            logger.info(json.dumps(to_log))

    def evaluate_and_log(self, model, context=None):
        output = self.evaluate(model)
        if get_distributed_context().is_main_process:
            self.log(output, context=context)
        return output
