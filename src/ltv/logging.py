import logging
from pathlib import Path
from .config import LOG_DIR

def get_logger(name: str = __name__, log_file: str = "pipeline.log") -> logging.Logger:
    logger = logging.getLogger(name)
    logger.setLevel(logging.DEBUG)

    if not logger.handlers:
        # Console handler (INFO)
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch_formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        ch.setFormatter(ch_formatter)

        # File handler (DEBUG)
        log_path = Path(LOG_DIR) / log_file
        fh = logging.FileHandler(log_path)
        fh.setLevel(logging.DEBUG)
        fh_formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        fh.setFormatter(fh_formatter)

        logger.addHandler(ch)
        logger.addHandler(fh)

    return logger
