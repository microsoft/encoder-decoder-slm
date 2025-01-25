import dataclasses
import logging
import shutil
from dataclasses import dataclass
from pathlib import Path

import torch

logger = logging.getLogger(__name__)


@dataclass
class SavedCheckpoint:
    checkpoint_path: Path
    score: float


class CheckpointSaver:
    def __init__(self, output_path, max_checkpoints_to_keep=3):
        self.checkpoint_dir = self._setup_checkpoint_dir(output_path)
        self.max_checkpoints_to_keep = max_checkpoints_to_keep

        self._best_checkpoints = []

    def _setup_checkpoint_dir(self, output_path):
        checkpoint_dir = output_path / "checkpoints"
        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        return checkpoint_dir

    def copy_best(self, output_path):
        if len(self._best_checkpoints) == 0:
            logger.info("No checkpoints to copy")
            return

        best_checkpoint = max(self._best_checkpoints, key=lambda c: c.score)
        logger.info(f"Copying best checkpoint {best_checkpoint.checkpoint_path} to {output_path}")
        shutil.copyfile(best_checkpoint.checkpoint_path, output_path)

    def save_if_best(self, model, global_step, score):
        """
        A high score is better.
        """
        if len(self._best_checkpoints) < self.max_checkpoints_to_keep:
            checkpoint_path = save_checkpoint(
                checkpoint_dir=self.checkpoint_dir,
                model=model,
                global_step=global_step,
                score=score,
            )
            self._best_checkpoints.append(SavedCheckpoint(checkpoint_path, score))
            logger.info(f"Saved checkpoint {checkpoint_path} with score {score}")
            return checkpoint_path

        worst_saved_score = min(c.score for c in self._best_checkpoints)
        if score > worst_saved_score:
            self._remove_worst_checkpoint()
            checkpoint_path = save_checkpoint(
                checkpoint_dir=self.checkpoint_dir,
                model=model,
                global_step=global_step,
                score=score,
            )
            self._best_checkpoints.append(SavedCheckpoint(checkpoint_path, score))
            logger.info(f"Saved checkpoint {checkpoint_path} with score {score}")
            return checkpoint_path

        logger.info(
            f"Did not save checkpoint with score {score} because"
            " it is not better than the worst saved score {worst_saved_score}"
        )

    def _remove_worst_checkpoint(self):
        worst_checkpoint = min(self._best_checkpoints, key=lambda c: c.score)
        if worst_checkpoint.checkpoint_path.exists():
            worst_checkpoint.checkpoint_path.unlink()
        self._best_checkpoints.remove(worst_checkpoint)
        logger.info(f"Deleted checkpoint {worst_checkpoint.checkpoint_path}")


def save_checkpoint(checkpoint_dir, model, global_step=None, score=None):
    if global_step is not None:
        checkpoint_path = checkpoint_dir / f"model_{global_step}.pt"
    else:
        checkpoint_path = checkpoint_dir / "model.pt"

    model_config = dataclasses.asdict(model.config)

    state = {
        "model_state_dict": model.state_dict(),
        "model_config": model_config,
    }
    if global_step is not None:
        state["global_step"] = global_step
    if score is not None:
        state["score"] = score
    torch.save(state, checkpoint_path)

    return checkpoint_path
