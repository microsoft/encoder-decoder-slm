import json
import logging
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import torch
import torch.nn.functional as F
from peft import LoraConfig, TaskType, get_peft_model
from torch import nn
from torch.utils.data import Dataset
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForCausalLM, HfArgumentParser

from mu.checkpoint import CheckpointSaver
from mu.distributed import distributed_manager, get_distributed_context
from mu.evaluate.generate import GenerateEvaluator
from mu.evaluate.loss import LossEvaluator
from mu.log import init_logging
from mu.models.modeling_mu import get_model
from mu.tokenizer import get_tokenizer
from mu.train import Trainer, get_dataloader, set_seeds

logger = logging.getLogger(__name__)


@contextmanager
def override_tokenizer_padding_side(tokenizer, padding_side):
    saved_padding_side = tokenizer.padding_side
    tokenizer.padding_side = padding_side
    try:
        yield tokenizer
    finally:
        tokenizer.padding_side = saved_padding_side


class Seq2SeqDataset(Dataset):
    def __init__(self, data_path: Path, tokenizer, encoder_seq_len: int, decoder_seq_len: int):
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.encoder_seq_len = encoder_seq_len
        self.decoder_seq_len = decoder_seq_len

        self._data = self._load_data(self.data_path)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(data_path={self.data_path}, "
            f"encoder_seq_len={self.encoder_seq_len}, "
            f"decoder_seq_len={self.decoder_seq_len})"
        )

    def _load_data(self, data_path: Path):
        data = [json.loads(line.strip()) for line in data_path.read_text().strip().split("\n")]
        return data

    def __len__(self):
        return len(self._data)

    def __getitem__(self, idx):
        item = self._data[idx]
        source_text = item["source"]
        target_text = item["target"]
        target_text = target_text + self.tokenizer.eos_token

        source_encodings = self.tokenizer(
            source_text,
            padding="max_length",
            truncation=True,
            max_length=self.encoder_seq_len,
            return_tensors="pt",
        )
        target_encodings = self.tokenizer(
            target_text,
            padding="max_length",
            truncation=True,
            max_length=self.decoder_seq_len,
            return_tensors="pt",
        )
        labels = target_encodings["input_ids"]

        decoder_input_ids = torch.zeros_like(labels)
        decoder_input_ids[:, 1:] = target_encodings["input_ids"][:, :-1]
        decoder_input_ids[:, 0] = self.tokenizer.bos_token_id

        decoder_attention_mask = torch.zeros_like(labels)
        decoder_attention_mask[:, 1:] = target_encodings["attention_mask"][:, :-1]
        decoder_attention_mask[:, 0] = 1

        sample = dict(
            encoder_input_ids=source_encodings["input_ids"].squeeze(0),
            encoder_attention_mask=source_encodings["attention_mask"].squeeze(0),
            decoder_input_ids=decoder_input_ids.squeeze(0),
            decoder_attention_mask=decoder_attention_mask.squeeze(0),
            targets=labels.squeeze(0),
        )

        return sample


class KDDataset(Dataset):
    def __init__(
        self, model, data_path: Path, tokenizer, encoder_seq_len: int, decoder_seq_len: int, temperature=0.9, top_k=128
    ):
        self.model = model
        self.data_path = data_path
        self.tokenizer = tokenizer
        self.encoder_seq_len = encoder_seq_len
        self.decoder_seq_len = decoder_seq_len
        self.temperature = temperature
        self.top_k = top_k

        self.seq2seq_dataset = Seq2SeqDataset(
            data_path=self.data_path,
            tokenizer=self.tokenizer,
            encoder_seq_len=self.encoder_seq_len,
            decoder_seq_len=self.decoder_seq_len,
        )

        self._data = self._load_data(self.data_path)

    def __repr__(self):
        return (
            f"{self.__class__.__name__}"
            f"(data_path={self.data_path}, "
            f"encoder_seq_len={self.encoder_seq_len}, "
            f"decoder_seq_len={self.decoder_seq_len})"
        )

    def _load_data(self, data_path: Path):
        data = [json.loads(line.strip()) for line in data_path.read_text().strip().split("\n")]
        return data

    def __len__(self):
        return len(self._data)

    def _generate_target(self, source_input_ids):
        distributed_context = get_distributed_context()

        device = distributed_context.device
        source_input_ids = source_input_ids.to(device)

        generated_ids = self.model.generate(
            input_ids=source_input_ids,
            max_new_tokens=self.decoder_seq_len,
            temperature=self.temperature,
            top_k=self.top_k,
            eos_token_id=self.tokenizer.eos_token_id,
            bos_token_id=self.tokenizer.bos_token_id,
        )
        generated_ids = generated_ids[:, 1:]  # remove bos token
        n_tokens = len(self.tokenizer)
        generated_ids[generated_ids > n_tokens] = self.tokenizer.unk_token_id
        assert generated_ids.size(0) == 1
        generated_ids = generated_ids[0]
        return generated_ids

    def _getitem_kd(self, idx):
        item = self._data[idx]
        source_text = item["source"]
        source_encodings = self.tokenizer(
            source_text,
            truncation=True,
            max_length=self.encoder_seq_len,
            return_tensors="pt",
        )
        source_input_ids = source_encodings["input_ids"]
        generated_ids = self._generate_target(source_input_ids)

        target_input_ids = torch.full((1, self.decoder_seq_len), self.tokenizer.pad_token_id, dtype=torch.long)
        target_input_ids[0, : len(generated_ids)] = generated_ids

        # if len(generated_ids) < self.decoder_seq_len:
        #     target_input_ids[0, len(generated_ids)] = self.tokenizer.eos_token_id

        target_attention_mask = (target_input_ids != self.tokenizer.pad_token_id).long()

        with override_tokenizer_padding_side(self.tokenizer, "left") as tokenizer:
            left_pad_source_encodings = tokenizer(
                source_text,
                padding="max_length",
                truncation=True,
                max_length=self.encoder_seq_len,
                return_tensors="pt",
            )

        teacher_input_ids = torch.cat([left_pad_source_encodings["input_ids"], target_input_ids], dim=1)
        teacher_attention_mask = torch.cat(
            [left_pad_source_encodings["attention_mask"], target_attention_mask],
            dim=1,
        )

        student_decoder_input_ids = torch.full_like(target_input_ids, self.tokenizer.pad_token_id)
        student_decoder_input_ids[:, 1:] = target_input_ids[:, :-1]
        student_decoder_input_ids[:, 0] = self.tokenizer.bos_token_id

        item = dict(
            teacher_input_ids=teacher_input_ids.squeeze(0),
            teacher_attention_mask=teacher_attention_mask.squeeze(0),
            student_decoder_input_ids=student_decoder_input_ids.squeeze(0),
        )
        return item

    def _getitem_seq2seq(self, idx):
        item = self.seq2seq_dataset[idx]
        return item

    def __getitem__(self, idx):
        item_seq2seq = self._getitem_seq2seq(idx)
        item_kd = self._getitem_kd(idx)
        item = dict(
            **item_seq2seq,
            **{f"kd_{k}": v for k, v in item_kd.items()},
        )
        return item


class KDModel(nn.Module):
    def __init__(
        self,
        student,
        teacher,
        tokenizer,
        alpha=1.0,
        encoder_seq_len=4096,
        decoder_seq_len=256,
    ):
        super().__init__()

        assert 0.0 <= alpha <= 1.0

        self.student = student
        self.teacher = teacher
        self.alpha = alpha
        self.tokenizer = tokenizer
        self.encoder_seq_len = encoder_seq_len
        self.decoder_seq_len = decoder_seq_len

    def _log_sample(self, **batch):
        threshold = 0.01
        should_log = random.random() <= threshold
        if should_log:
            decoded_items = {
                k: self.tokenizer.decode(v[-1][v[-1] != -100], skip_special_tokens=True) for k, v in batch.items()
            }
            logger.info(
                json.dumps(
                    {
                        "message": "kd_sample",
                        **decoded_items,
                    }
                )
            )

    def train(self, mode=True):
        self.student.train(mode)
        self.teacher.eval()

    def forward_kd(
        self,
        encoder_input_ids,
        decoder_input_ids,
        teacher_input_ids,
        teacher_attention_mask,
    ):
        student_outputs = self.student(encoder_input_ids=encoder_input_ids, decoder_input_ids=decoder_input_ids)
        student_logits = student_outputs.logits

        with torch.no_grad():
            teacher_outputs = self.teacher(input_ids=teacher_input_ids, attention_mask=teacher_attention_mask)
        teacher_logits = teacher_outputs.logits
        teacher_logits_target = teacher_logits[
            :, self.encoder_seq_len - 1 : self.encoder_seq_len - 1 + self.decoder_seq_len, :
        ]

        min_vocab_size = min(student_logits.size(-1), teacher_logits_target.size(-1))
        student_logits = student_logits[:, :, :min_vocab_size]
        teacher_logits_target = teacher_logits_target[:, :, :min_vocab_size]

        student_logits_flat = student_logits.reshape(-1, min_vocab_size)
        teacher_logits_target_flat = teacher_logits_target.reshape(-1, min_vocab_size)

        loss = F.kl_div(
            F.log_softmax(student_logits_flat, dim=-1),
            F.log_softmax(teacher_logits_target_flat, dim=-1),
            reduction="batchmean",
            log_target=True,
        )

        outputs = SimpleNamespace(loss=loss)
        return outputs

    def forward(
        self,
        encoder_input_ids,
        decoder_input_ids,
        kd_teacher_input_ids,
        kd_teacher_attention_mask,
        kd_student_decoder_input_ids,
        targets=None,
        encoder_attention_mask=None,
        decoder_attention_mask=None,
    ):
        self._log_sample(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            targets=targets,
            kd_teacher_input_ids=kd_teacher_input_ids,
            kd_student_decoder_input_ids=kd_student_decoder_input_ids,
        )

        kd_outputs = self.forward_kd(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=kd_student_decoder_input_ids,
            teacher_input_ids=kd_teacher_input_ids,
            teacher_attention_mask=kd_teacher_attention_mask,
        )
        kd_loss = kd_outputs.loss

        outputs = self.student(
            encoder_input_ids=encoder_input_ids,
            decoder_input_ids=decoder_input_ids,
            encoder_attention_mask=encoder_attention_mask,
            decoder_attention_mask=decoder_attention_mask,
            targets=targets,
        )
        ce_loss = outputs.loss
        loss = self.alpha * kd_loss + (1.0 - self.alpha) * ce_loss

        outputs = SimpleNamespace(
            loss=loss,
            kd_loss=kd_loss,
            ce_loss=ce_loss,
        )
        return outputs


@dataclass
class TrainText2TextByKD:
    student_path: Path = Path("artifacts/models/model.pt")
    teacher_path: Path = Path("artifacts/models/teacher.pt")
    train_data_path: Path = Path("artifacts/data/train.jsonl")
    val_data_path: Path = Path("artifacts/data/val.jsonl")
    encoder_seq_len: int = 1024
    decoder_seq_len: int = 256
    batch_size: int = 6
    global_batch_size: int = 128
    num_epochs: int = 128
    learning_rate: float = 1e-4
    weight_decay: float = 0.01
    warmup_steps: int = 4
    max_grad_norm: float = 1.0
    generate_temperature: float = 0.9
    generate_top_k: int = 128
    generate_frequency: int = 32
    val_loss_frequency: int = 32
    kd_alpha: float = 0.9
    output_path: Path = Path("outputs")

    def get_teacher(self, teacher_name: str = "microsoft/Phi-3.5-mini-instruct"):
        config = AutoConfig.from_pretrained(teacher_name)
        teacher = AutoModelForCausalLM.from_config(config)

        peft_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,
            lora_alpha=32,
            lora_dropout=0.05,
            bias="none",
            target_modules=["o_proj", "qkv_proj", "gate_up_proj", "down_proj"],
        )
        teacher = get_peft_model(teacher, peft_config)

        teacher_loaded = torch.load(self.teacher_path, map_location=self.device, weights_only=True)
        teacher_state_dict = teacher_loaded["model_state_dict"]
        teacher.load_state_dict(teacher_state_dict)
        teacher = teacher.merge_and_unload()

        teacher.eval()
        teacher = teacher.to(torch.bfloat16).to(self.device)
        for param in teacher.parameters():
            param.requires_grad = False
        return teacher

    def setup(self):
        set_seeds()

        distributed_context = get_distributed_context()
        self.device = distributed_context.device

        self.tokenizer = get_tokenizer()

        self.student = get_model(self.student_path, device=self.device)
        self.teacher = self.get_teacher()

        self.train_dataset = KDDataset(
            model=self.student,
            data_path=self.train_data_path,
            tokenizer=self.tokenizer,
            encoder_seq_len=self.encoder_seq_len,
            decoder_seq_len=self.decoder_seq_len,
        )

        self.train_dataloader = get_dataloader(
            self.train_dataset, batch_size=self.batch_size, shuffle=True, drop_last=True
        )
        data_steps = int(len(self.train_dataloader)) * self.num_epochs
        logger.info(f"Will train for {data_steps=}")

        kd_model = KDModel(
            student=self.student,
            teacher=self.teacher,
            tokenizer=self.tokenizer,
            alpha=self.kd_alpha,
            encoder_seq_len=self.encoder_seq_len,
            decoder_seq_len=self.decoder_seq_len,
        )

        self.trainer = Trainer(
            model=kd_model,
            data_steps=data_steps,
            batch_size=self.batch_size,
            max_grad_norm=self.max_grad_norm,
            global_batch_size=self.global_batch_size,
            weight_decay=self.weight_decay,
            learning_rate=self.learning_rate,
            warmup_steps=self.warmup_steps,
        )

        generate_dataset = Seq2SeqDataset(
            data_path=self.val_data_path,
            tokenizer=self.tokenizer,
            encoder_seq_len=self.encoder_seq_len,
            decoder_seq_len=self.decoder_seq_len,
        )
        generate_dataloader = get_dataloader(generate_dataset, batch_size=1, shuffle=False, drop_last=False)
        self.generate_evaluator = GenerateEvaluator(
            dataloader=generate_dataloader,
            tokenizer=self.tokenizer,
            max_new_tokens=self.decoder_seq_len,
            temperature=self.generate_temperature,
            top_k=self.generate_top_k,
        )

        val_dataset = KDDataset(
            model=self.student,
            data_path=self.val_data_path,
            tokenizer=self.tokenizer,
            encoder_seq_len=self.encoder_seq_len,
            decoder_seq_len=self.decoder_seq_len,
        )
        val_dataloader = get_dataloader(val_dataset, batch_size=self.batch_size, shuffle=True, drop_last=False)
        self.val_loss_evaluator = LossEvaluator(dataname="val", dataloader=val_dataloader)

        self.checkpoint_saver = CheckpointSaver(self.output_path)

    def train(self):
        is_main_process = get_distributed_context().is_main_process
        for epoch in range(self.num_epochs):
            progress_bar = tqdm(self.train_dataloader, desc=f"Epoch {epoch + 1}/{self.num_epochs}")

            for step, batch in enumerate(progress_bar):
                global_step = epoch * len(self.train_dataloader) + step
                step_context = dict(
                    epoch_step=step,
                    global_step=global_step,
                    epoch=epoch,
                )

                _, metrics = self.trainer.step(batch)

                if is_main_process and metrics is not None:
                    self.trainer.log(metrics, context=step_context)

                if (global_step + 1) % self.generate_frequency == 0:
                    self.generate_evaluator.evaluate_and_log(self.student, context=step_context)

                if (global_step + 1) % self.val_loss_frequency == 0:
                    val_loss_outputs = self.val_loss_evaluator.evaluate_and_log(
                        self.trainer.model, context=step_context
                    )
                    val_loss = val_loss_outputs.loss

                    if is_main_process:
                        self.checkpoint_saver.save_if_best(self.student, global_step, score=-val_loss)

    def __call__(self):
        self.setup()
        self.train()


def setup_logging():
    log_path = Path("outputs") / "logs"
    init_logging(log_path)


def main():
    setup_logging()

    parser = HfArgumentParser(TrainText2TextByKD)
    train_text2text_by_kd = parser.parse_args_into_dataclasses()[0]

    distributed_context = get_distributed_context()

    if distributed_context.is_distributed:
        with distributed_manager():
            train_text2text_by_kd()
    else:
        train_text2text_by_kd()


if __name__ == "__main__":
    main()
