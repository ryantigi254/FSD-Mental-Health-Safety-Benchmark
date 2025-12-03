"""Logging configuration for the benchmark."""

import logging
import sys
from pathlib import Path
from datetime import datetime


def setup_logging(log_dir: str = "logs", level: int = logging.INFO):
    """Configure logging for the benchmark."""
    Path(log_dir).mkdir(exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = Path(log_dir) / f"evaluation_{timestamp}.log"

    logging.basicConfig(
        level=level,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler(sys.stdout),
        ],
    )

    logging.getLogger("transformers").setLevel(logging.WARNING)
    logging.getLogger("urllib3").setLevel(logging.WARNING)

    logger = logging.getLogger(__name__)
    logger.info(f"Logging initialised. Log file: {log_file}")

    return logger


