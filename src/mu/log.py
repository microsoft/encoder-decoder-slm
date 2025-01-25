import logging
from pathlib import Path


def init_logging(log_dir: Path):
    stdout_handler = logging.StreamHandler()
    log_path = log_dir / "log.txt"
    log_path.parent.mkdir(parents=True, exist_ok=True)
    file_handler = logging.FileHandler(log_path)
    logging.basicConfig(level=logging.INFO, handlers=[stdout_handler, file_handler])
