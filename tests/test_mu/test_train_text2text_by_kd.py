import json
import re
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

import torch

from mu.checkpoint import save_checkpoint
from mu.models.modeling_mu import Mu, MuConfig
from mu.train_text2text_by_kd import TrainText2TextByKD

from .torchrun import torchrun


def _get_data(n_samples=4):
    question = "What is the capital of the United States?"
    context = "Washington, D.C. is the capital of the United States."
    answer = "Washington, D.C."

    question_template = f"""Answer the following question based only on the context.
###Question
{question}"""
    context_template = f"""
###Context
{context}"""
    answer_postfix = "\n###Answer\n"

    source = question_template + context_template + answer_postfix
    target = answer

    item = dict(source=source, target=target)
    data = [item] * 4
    return data


def _write_data(data, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    data_text = "\n".join([json.dumps(item) for item in data])
    path.write_text(data_text)
    return path


def _get_student_model(device="cpu"):
    config = MuConfig()
    model = Mu(config, device=device)
    model = model.to(dtype=torch.bfloat16).to(device=device)
    return model


def _write_student_model(model, output_dir):
    checkpoint_path = save_checkpoint(output_dir, model)
    return checkpoint_path


def _run_train_overfit(
    student_path: Path, train_data_path: Path, output_path: Path, kd_alpha: float = 1.0, num_epochs: int = 512
):
    train_text2text_by_kd = TrainText2TextByKD(
        student_path=student_path,
        teacher_path="artifacts/models/teacher.pt",
        train_data_path=train_data_path,
        val_data_path=train_data_path,
        encoder_seq_len=128,
        decoder_seq_len=32,
        batch_size=1,
        global_batch_size=4,
        num_epochs=num_epochs,
        learning_rate=1e-4,
        warmup_steps=1,
        generate_frequency=32,
        val_loss_frequency=32,
        kd_alpha=kd_alpha,
        output_path=output_path,
    )
    train_text2text_by_kd()


class TestKDOverfit(unittest.TestCase):
    def test_kd_overfit(self):
        kd_alpha = 1.0
        num_epochs = 512
        expected_loss = 0.1
        expected_generations = [
            "The capital of the United States is Washington, D.C.",
            "Washington, D.C.",
            "Washington, D.C. is the capital of the United States.",
        ]
        self._test_overfit(
            kd_alpha=kd_alpha,
            num_epochs=num_epochs,
            expected_generations=expected_generations,
            expected_loss=expected_loss,
        )

    def test_ce_overfit(self):
        kd_alpha = 0.0
        num_epochs = 128
        expected_loss = 0.01
        expected_generations = [
            "Washington, D.C.",
        ]
        self._test_overfit(
            kd_alpha=kd_alpha,
            num_epochs=num_epochs,
            expected_generations=expected_generations,
            expected_loss=expected_loss,
        )

    def _get_log_lines(self, rundir, pattern=None):
        log_path = rundir / "rank=0/log.txt"
        log_lines = log_path.read_text().splitlines()
        if pattern is not None:
            log_lines = [line for line in log_lines if pattern in line]
        return log_lines

    def _parse_log_line(self, log_line):
        json_line = json.loads(re.sub(r"(INFO|DEBUG):[a-z_.]+:", "", log_line))
        return json_line

    def _assert_loss(self, rundir, expected_loss):
        log_lines = self._get_log_lines(rundir, pattern="loss_evaluator_metrics")
        loss_messages = [self._parse_log_line(line) for line in log_lines]
        n_last_losses = 3
        avg_last_loss = torch.tensor([message["loss"] for message in loss_messages[-n_last_losses:]]).mean().item()
        self.assertLessEqual(avg_last_loss, expected_loss)

    def _assert_generations(self, rundir, expected_generations):
        log_lines = self._get_log_lines(rundir, pattern="generate_evaluation")
        generation_messages = [self._parse_log_line(line) for line in log_lines]
        generations = [message["generated"] for message in generation_messages]
        n_last_generations = 5
        last_generations = generations[-n_last_generations:]
        n_right_in_last = sum(generation in expected_generations for generation in last_generations)
        self.assertGreaterEqual(n_right_in_last, 4)

    def _test_overfit(self, expected_loss, expected_generations, kd_alpha=1.0, num_epochs=512):
        with TemporaryDirectory(prefix="test_mu") as rundir:
            rundir = Path(rundir)

            train_data_path = rundir / "train_data.jsonl"
            train_data = _get_data()
            train_data_path = _write_data(train_data, train_data_path)

            output_path = rundir / "output"

            model = _get_student_model()
            student_path = _write_student_model(model, rundir)

            function = _run_train_overfit
            kwargs = dict(
                student_path=student_path,
                train_data_path=train_data_path,
                output_path=output_path,
                kd_alpha=kd_alpha,
                num_epochs=num_epochs,
            )
            torchrun(rundir, function, kwargs=kwargs)

            self._assert_loss(rundir, expected_loss)
            self._assert_generations(rundir, expected_generations)
